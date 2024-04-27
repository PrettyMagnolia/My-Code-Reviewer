import os
import torch
import logging
import argparse
import random
import json
from tqdm import tqdm
import multiprocessing
import time
from itertools import cycle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from configs import add_args, set_seed, set_dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils import CommentGenDataset, SimpleGenDataset
from evaluator.smooth_bleu import bleu_fromstr
import wandb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

exp_name = "exp" + time.strftime("-%Y%m%d-%H_%M_%S", time.localtime(int(round(time.time() * 1000)) / 1000))
wandb_project_name = "My_Code_Reviewer"


def setup_wandb(config):
    print('setup_wandb...')
    # 初始化wandb
    wandb.init(config=config, project=wandb_project_name, name=exp_name)


def get_loaders(data_files, args, tokenizer, pool, eval=False):
    def fn(features):
        return features

    global_rank = args.global_rank
    for data_file in data_files:
        if args.raw_input:
            dataset = SimpleGenDataset(tokenizer, pool, args, data_file)
        else:
            dataset = CommentGenDataset(tokenizer, pool, args, data_file)
        data_len = len(dataset)
        if global_rank == 0:
            logger.info(f"Data length: {data_len}.")
        if eval:
            sampler = SequentialSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size if not eval else args.eval_batch_size, \
                                num_workers=args.cpu_count, collate_fn=fn)
        yield dataset, sampler, dataloader


def eval_bleu_epoch(args, eval_dataloader, model, tokenizer):
    logger.info(f"  ***** Running bleu evaluation on {args.eval_file} *****")
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    pred_ids, ex_ids = [], []
    for step, examples in enumerate(eval_dataloader, 1):
        source_ids = torch.tensor(
            [ex.source_ids for ex in examples], dtype=torch.long
        ).to(args.local_rank)
        ids = [ex.example_id for ex in examples]
        source_mask = source_ids.ne(tokenizer.pad_id)
        preds = model.generate(source_ids,
                               attention_mask=source_mask,
                               use_cache=True,
                               num_beams=args.beam_size,
                               early_stopping=True,
                               max_length=args.max_target_length)
        top_preds = list(preds.cpu().numpy())
        pred_ids.extend(top_preds)
    # [1:] to remove beginning '<msg>'
    pred_nls = [tokenizer.decode(id[1:], skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    valid_file = args.dev_filename
    golds = []
    with open(valid_file, "r") as f:
        for line in f:
            golds.append(json.loads(line)["msg"])
    golds = golds[:len(pred_nls)]
    bleu = bleu_fromstr(pred_nls, golds, rmstop=False)
    return bleu


def save_model(model, optimizer, scheduler, output_dir, config):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    config.save_pretrained(output_dir)
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)
    output_optimizer_file = os.path.join(output_dir, "optimizer.pt")
    torch.save(
        optimizer.state_dict(),
        output_optimizer_file,
        _use_new_zipfile_serialization=False,
    )
    output_scheduler_file = os.path.join(output_dir, "scheduler.pt")
    torch.save(
        scheduler.state_dict(),
        output_scheduler_file,
        _use_new_zipfile_serialization=False,
    )


def main(args):
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank() % args.gpu_per_node
    args.global_rank = local_rank + args.node_index * args.gpu_per_node
    args.local_rank = local_rank
    args.world_size = dist.get_world_size()
    logger.warning("Process rank: %s, global rank: %s, world size: %s, bs: %s",
                   args.local_rank, args.global_rank, \
                   torch.distributed.get_world_size(), \
                   args.train_batch_size)
    torch.cuda.set_device(local_rank)

    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)

    # 使用 DDP（DistributedDataParallel）将模型移动到 GPU 上，并设置多 GPU 训练
    model = DDP(model.cuda(), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 创建多进程池
    pool = multiprocessing.Pool(args.cpu_count)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    # 使用 AdamW 优化器
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    # 设置 warmup 步数为总训练步数的 10%
    args.warmup_steps = int(args.train_steps * 0.1)
    # 使用 linear warmup 和 decay 的调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
    )

    # 检查是否存在最新的 optimizer 和 scheduler 的检查点文件，如果存在则加载它们
    if os.path.exists("{}/checkpoints-last/optimizer.pt".format(args.output_dir)):
        optimizer.load_state_dict(
            torch.load(
                "{}/checkpoints-last/optimizer.pt".format(args.output_dir),
                map_location="cpu",
            )
        )
        scheduler.load_state_dict(
            torch.load(
                "{}/checkpoints-last/scheduler.pt".format(args.output_dir),
                map_location="cpu",
            )
        )
    # 初始化全局步数为 0，并设置保存模型的步数
    global_step = 0
    save_steps = args.save_steps

    # 获取训练数据文件名和验证数据文件名
    train_file = args.train_filename
    valid_file = args.dev_filename

    # 如果训练文件是一个目录，则获取目录下所有符合条件的文件名
    if os.path.isdir(train_file):
        train_files = [file for file in os.listdir(train_file) if file.startswith("train") and file.endswith(".jsonl")]
    else:
        train_files = [train_file]

    # 随机打乱训练文件列表，并添加文件路径前缀
    random.seed(args.seed)
    random.shuffle(train_files)
    train_files = [os.path.join(train_file, file) for file in train_files]

    # 设置验证文件列表
    valid_files = [valid_file]
    # bleu = eval_bleu_epoch(args, valid_dataloader, model, tokenizer)
    # logger.warning("Initial bleu: {}".format(bleu))

    # 遍历每个训练周期
    for epoch in range(1, args.train_epochs + 1):
        # set seed for reproducible data split
        save_seed = args.seed
        args.seed += epoch
        set_seed(args)
        args.seed = save_seed

        # 设置模型为训练模式
        model.train()
        nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
        for _, _, train_dataloader in get_loaders(train_files, args, tokenizer, pool):  # WARNING: this is an iterator, to save memory
            for step, examples in enumerate(train_dataloader, 1):
                if step == 1:
                    # 打印第一个批次的信息
                    ex = examples[0]
                    # logger.info(f"batch size: {len(examples)}")
                    # logger.info(f"example source: {tokenizer.convert_ids_to_tokens(ex.source_ids)}")
                    # logger.info(f"example label: {tokenizer.convert_ids_to_tokens(ex.source_labels)}")
                    # logger.info(f"example target: {tokenizer.convert_ids_to_tokens(ex.target_ids)}")

                # 将数据转换为张量并移到指定的设备
                source_ids = torch.tensor(
                    [ex.source_ids for ex in examples], dtype=torch.long
                ).to(local_rank)
                source_labels = None
                target_ids = torch.tensor(
                    [ex.target_ids for ex in examples], dtype=torch.long
                ).to(local_rank)
                source_mask = source_ids.ne(tokenizer.pad_id)
                target_mask = target_ids.ne(tokenizer.pad_id)

                # 计算模型的损失
                loss = model(
                    input_ids=source_ids,
                    input_labels=source_labels,
                    decoder_input_ids=target_ids,
                    attention_mask=source_mask,
                    decoder_attention_mask=target_mask,
                    encoder_loss=False
                )

                if args.gpu_per_node > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                # 当累积梯度步数达到设定值时，更新参数
                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    # 如果是全局排名为 0 的进程，并且达到了日志步数，则打印训练损失信息
                    if args.global_rank == 0 and global_step % args.log_steps == 0:
                        train_loss = round(
                            tr_loss * args.gradient_accumulation_steps / nb_tr_steps,
                            4,
                        )
                        logger.info(
                            "step {}/{}: Train loss {}".format(
                                global_step,
                                args.train_steps,
                                round(train_loss, 3),
                            )
                        )
                # 如果全局步数达到了训练步数，则进行验证并保存模型
                if global_step == args.train_steps and args.global_rank == 0:
                    # end training
                    _, _, valid_dataloader = next(get_loaders(valid_files, args, tokenizer, pool, eval=True))
                    bleu = eval_bleu_epoch(args, valid_dataloader, model, tokenizer)
                    output_dir = os.path.join(args.output_dir, "checkpoints-last" + "-" + str(bleu))
                    save_model(model, optimizer, scheduler, output_dir, config)
                    logger.info(f"Reach max steps {args.train_steps}.")
                    time.sleep(5)
                    return
                # 如果全局排名为 0 的进程，且满足保存步数条件，则进行验证并保存模型
                if args.global_rank == 0 and \
                        global_step % save_steps == 0 and \
                        nb_tr_steps % args.gradient_accumulation_steps == 0:
                    _, _, valid_dataloader = next(get_loaders(valid_files, args, tokenizer, pool, eval=True))
                    bleu = eval_bleu_epoch(args, valid_dataloader, model, tokenizer)
                    output_dir = os.path.join(args.output_dir, "checkpoints-" + str(global_step) + "-" + str(bleu))
                    save_model(model, optimizer, scheduler, output_dir, config)
                    logger.info(
                        "Save the {}-step model and optimizer into {}".format(
                            global_step, output_dir
                        )
                    )
                    time.sleep(5)
        # 输出每个epoch的平均loss
        avg_epoch_loss = tr_loss / nb_tr_steps
        logger.info(f"Epoch {epoch}: Average Loss = {avg_epoch_loss}")
        wandb.log({"Epoch Loss": avg_epoch_loss}, step=epoch)

        # 判断是否到了指定的epoch，保存模型
        if epoch % args.save_interval_epochs == 0 or epoch == args.train_epochs:
            _, _, valid_dataloader = next(get_loaders(valid_files, args, tokenizer, pool, eval=True))
            bleu = eval_bleu_epoch(args, valid_dataloader, model, tokenizer)
            output_dir = os.path.join(args.output_dir, "checkpoints-" + str(global_step) + "-" + str(bleu))
            save_model(model, optimizer, scheduler, output_dir, config)
            logger.info(
                "Save the {}-step model and optimizer into {}".format(
                    global_step, output_dir
                )
            )
            time.sleep(5)


if __name__ == "__main__":
    # 手动设置全局变量
    mnt_dir = "/home/codereview"

    MASTER_HOST = "localhost"
    MASTER_PORT = 23333
    RANK = 0
    PER_NODE_GPU = 1
    WORLD_SIZE = 1
    NODES = 1
    NCCL_DEBUG = "INFO"

    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(WORLD_SIZE)
    os.environ['MASTER_ADDR'] = MASTER_HOST
    os.environ['MASTER_PORT'] = str(MASTER_PORT)
    os.environ['NCCL_DEBUG'] = NCCL_DEBUG

    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.cpu_count = multiprocessing.cpu_count()
    # remove long tokenization warning. ref: https://github.com/huggingface/transformers/issues/991

    args.train_epochs = 30
    args.model_name_or_path = "/data/lyf/code/Code_Reviewer/3_Pretrained_Model"
    args.output_dir = "/data/lyf/code/Code_Reviewer/0_Result"
    args.train_filename = "/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/msg-train-small.jsonl"
    args.dev_filename = "/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/msg-valid-small.jsonl"
    args.max_source_length = 512
    args.max_target_length = 128
    args.train_batch_size = 6
    args.learning_rate = 3e-4
    args.gradient_accumulation_steps = 3
    args.mask_rate = 0.15
    args.save_steps = 1800
    args.log_steps = 100
    args.train_steps = 60000
    args.gpu_per_node = PER_NODE_GPU
    args.node_index = RANK
    args.seed = 2233
    args.raw_input = True

    args.save_interval_epochs = 100

    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logger.info(args)
    # 初始化wandb
    setup_wandb(args)
    # 开始训练
    main(args)
    # 结束wandb
    wandb.finish()
    logger.info("Training finished.")
    # torch.multiprocessing.spawn(main, args=(args,), nprocs=torch.cuda.device_count())
