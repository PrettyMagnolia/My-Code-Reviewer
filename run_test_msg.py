import os, json
import torch
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
from itertools import cycle
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from configs import add_args, set_seed, set_dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils import CommentGenDataset, SimpleGenDataset
from evaluator.smooth_bleu import bleu_fromstr


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_loader(data_file, args, tokenizer, pool):
    def fn(features):
        return features
    logger.info(f"Start data file {data_file}.")
    if args.raw_input:
        dataset = SimpleGenDataset(tokenizer, pool, args, data_file)
    else:
        dataset = CommentGenDataset(tokenizer, pool, args, data_file)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=0, collate_fn=fn)
    logger.info(f"Finish data files {data_file}.")
    return dataset, sampler, dataloader


def eval_epoch_bleu(args, eval_dataloader, model, tokenizer):
    """
    在评估集上运行 BLEU 评估。

    Args:
        args (argparse.Namespace): 命令行参数对象。
        eval_dataloader (DataLoader): 评估集的数据加载器。
        model (nn.Module): 待评估的模型。
        tokenizer (Tokenizer): 用于处理文本数据的分词器。

    Returns:
        float: BLEU 分数。
    """
    logger.info(f"  ***** Running bleu evaluation on {args.eval_file} *****")
    logger.info("  Batch size = %d", args.eval_batch_size)

    # 将模型设置为评估模式
    model.eval()
    if hasattr(model, "module"):
        model = model.module

    pred_ids, ex_ids = [], []

    # 遍历评估数据加载器，生成预测结果
    for step, examples in tqdm(enumerate(eval_dataloader, 1)):
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

    # 解码预测结果和参考答案
    pred_nls = [tokenizer.decode(id[2:], skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    valid_file = args.eval_file
    golds = []
    with open(valid_file, "r") as f:
        for line in f:
            golds.append(json.loads(line)["msg"])
    golds = golds[:len(pred_nls)]

    # 将预测结果和参考答案写入文件
    with open(os.path.join(args.model_name_or_path, "preds.txt"), "w", encoding="utf-8") as f:
        for pred in pred_nls:
            f.write(pred.strip() + "\n")
    with open(os.path.join(args.model_name_or_path, "golds.txt"), "w", encoding="utf-8") as f:
        for gold in golds:
            f.write(gold.strip() + "\n")

    # 计算 BLEU 分数
    bleu = bleu_fromstr(pred_nls, golds, rmstop=False)
    logger.warning(f"WithStop BLEU: {bleu}")
    bleu = bleu_fromstr(pred_nls, golds, rmstop=True)
    return bleu


def main(args):
    """
    主函数，用于启动训练和评估流程。

    Args:
        args (argparse.Namespace): 命令行参数对象。
    """
    # 初始化进程组
    dist.init_process_group(backend="gloo")

    # 获取本地和全局排名
    local_rank = dist.get_rank() % args.gpu_per_node
    args.global_rank = local_rank + args.node_index * args.gpu_per_node
    args.local_rank = local_rank
    args.world_size = dist.get_world_size()
    logger.warning("Process rank: %s, global rank: %s, world size: %s, bs: %s",
                   args.local_rank, args.global_rank, \
                   torch.distributed.get_world_size(), \
                   args.eval_batch_size)
    torch.cuda.set_device(local_rank)

    # 设置种子以及构建或加载生成模型
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model = DDP(model.cuda(), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    pool = multiprocessing.Pool(args.cpu_count)
    data_file = args.eval_file
    set_seed(args)
    _, _, dataloader = get_loader(data_file, args, tokenizer, pool)  # WARNING: this is an iterator, to save memory

    # 将模型设置为评估模式，并在评估集上进行 BLEU 评估
    model.eval()
    bleu = eval_epoch_bleu(args, dataloader, model, tokenizer)
    logger.warning(f"BLEU: {bleu}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.cpu_count = multiprocessing.cpu_count()
    # remove long tokenization warning. ref: https://github.com/huggingface/transformers/issues/991
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logger.info(args)
    main(args)
    logger.info("Test finished.")
    # torch.multiprocessing.spawn(main, args=(args,), nprocs=torch.cuda.device_count())
