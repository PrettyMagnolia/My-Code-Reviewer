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
from utils import CommentGenDataset, SimpleGenDataset, attention_plot, read_stopwords, filter_stopwords, top_k_token_dict, merge_dict
from evaluator.smooth_bleu import bleu_fromstr
import torch.nn.functional as F

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
    focus = []

    # 遍历评估数据加载器，生成预测结果
    for step, examples in tqdm(enumerate(eval_dataloader, 1)):
        source_ids = torch.tensor(
            [ex.source_ids for ex in examples], dtype=torch.long
        ).to(args.local_rank)
        ids = [ex.example_id for ex in examples]
        source_mask = source_ids.ne(tokenizer.pad_id)
        outputs = model.generate(source_ids,
                                 attention_mask=source_mask,
                                 use_cache=True,
                                 # num_beams=args.beam_size,
                                 # early_stopping=True,
                                 do_sample=True,
                                 top_k=args.topk,
                                 max_length=args.max_target_length,
                                 output_attentions=True,
                                 output_hidden_states=True,
                                 output_scores=True,
                                 return_dict_in_generate=True)
        preds = outputs.sequences
        logits = outputs.scores
        cross_attentions = outputs.cross_attentions
        encoder_attentions = outputs.encoder_attentions
        decoder_attentions = outputs.decoder_attentions

        input_tokens = [tokenizer.decode(token_id, clean_up_tokenization_spaces=False) for token_id in list(source_ids[0].cpu().numpy())]

        # 解码并记录特殊标记的下标
        special_token_indices = []
        decoded_tokens = []
        for i, token_id in enumerate(list(source_ids[0].cpu().numpy())):
            token = tokenizer.decode(token_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if token.startswith("<") and token.endswith(">"):  # 检查是否为特殊标记
                special_token_indices.append(i)
            else:
                decoded_tokens.append(token)
        input_tokens = decoded_tokens

        res_focus_dict = {}

        # 获取预测结果的长度 前两位为起始标志 因此真正的序列长度要-2
        # if args.has_focus:
        #     seq_len = preds.size(1) - 2
        #     # 使用贪婪搜索时 cross_attention的长度为 seq_len+1
        #     for i in range(len(cross_attentions)):
        #         output_tokens = [
        #             tokenizer.decode([preds[0][i + 1]], skip_special_tokens=True, clean_up_tokenization_spaces=False)]
        #         # 生成第i个token时注意力的情况
        #         cur_attentions = cross_attentions[i]
        #         for j in range(len(cur_attentions)):
        #             # 第j层的注意力
        #             token_name = tokenizer.decode(preds[0][i + 1], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        #             for k in range(cur_attentions[j].size(1)):
        #                 # print(cur_attentions[j][0, k, :])
        #                 output_tensor = cur_attentions[j][0, k, :]
        #                 all_indices = list(range(output_tensor.size(1)))
        #                 indices_to_keep = torch.tensor(list(set(all_indices) - set(special_token_indices))).to(
        #                     args.local_rank)
        #                 # 选择不在删除索引列表中的元素
        #                 new_tensor = torch.index_select(output_tensor, 1, indices_to_keep).to(args.local_rank)
        #                 # 将token_id的注意力分数从大到小排序
        #                 top_values, top_indices = torch.topk(new_tensor, new_tensor.size(1))
        #
        #                 # 获取token列表和对应的注意力分数列表
        #                 top_tokens = [input_tokens[idx] for idx in top_indices[0].cpu().numpy()]
        #                 top_values = [top_value for top_value in top_values[0].cpu().numpy()]
        #
        #                 # 获取前k个token和对应的注意力分数字典
        #                 top_tokens_dict = top_k_token_dict(top_tokens, top_values, k=args.focus_len)
        #
        #                 # 合并字典
        #                 res_focus_dict = merge_dict(res_focus_dict, top_tokens_dict)
        #
        #                 # # 获取
        #                 # print(f'layer{j + 1}: bert_attention_weight_head_{k + 1}:', top_tokens_dict)
        #                 # # attention 归一化
        #                 # attentions_norm = F.normalize(top_values, p=2, dim=1)
        #                 # print(f'layer{j + 1}: bert_attention_weight_head_{k + 1}:', attentions_norm.cpu().numpy())
        #                 # # 显示第ii个Head的Attention
        #                 # attention_plot(attentions_norm.cpu().numpy(), annot=True,
        #                 #                x_texts=top_tokens,
        #                 #                y_texts=output_tokens, figsize=(15, 15),
        #                 #                figure_path='./figures',
        #                 #                figure_name='layer{}bert_attention_weight_head_{}.png'.format(j + 1, k + 1))
        #     # 按照注意力分数从大到小排序 取前k个最高的key
        #     res_focus = sorted(res_focus_dict, key=res_focus_dict.get, reverse=True)[:args.focus_len]
        #     # focus.append(res_focus)
        #
        #     # 将模型的关注点写入文件
        #     print("输入为：", input_tokens, "\n注意力为：", res_focus)
        #     with open(os.path.join(args.output_dir, args.focus_file_name), "a", encoding="utf-8") as f:
        #         foc_str = ', '.join(res_focus)
        #         f.write(foc_str + "\n")

        top_preds = list(preds.cpu().numpy())

        probs = [torch.softmax(log, dim=-1) for log in logits]

        for i, token_id in enumerate(top_preds[0][2:]):
            token_nls = tokenizer.decode(token_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            token_pron = probs[i][0, token_id].item()
            # print(f"Token ID: {token_id}, Token nls: {token_nls}, Probability: {token_pron}")
        pred_ids.extend(top_preds)

    # 解码预测结果和参考答案
    pred_nls = [tokenizer.decode(id[2:], skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in
                pred_ids]
    valid_file = args.eval_file
    golds = []
    with open(valid_file, "r") as f:
        for line in f:
            golds.append(json.loads(line)["msg"])
    golds = golds[:len(pred_nls)]

    # 将预测结果和参考答案写入文件
    with open(os.path.join(args.output_dir, "preds_origin_topk{}.txt".format(args.topk)), "w", encoding="utf-8") as f:
        for pred in pred_nls:
            f.write(pred.strip() + "\n")
    # with open(os.path.join(args.output_dir, "golds_{}.txt".format(args.check)), "w", encoding="utf-8") as f:
    #     for gold in golds:
    #         f.write(gold.strip() + "\n")

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
    dist.init_process_group(backend="nccl")

    # 获取本地和全局排名
    local_rank = dist.get_rank() % args.gpu_per_node
    # todo 修改使用的gpu号
    local_rank = 1

    args.global_rank = local_rank + args.node_index * args.gpu_per_node
    args.local_rank = local_rank
    args.world_size = dist.get_world_size()
    logger.warning("Process rank: %s, global rank: %s, world size: %s, bs: %s",
                   args.local_rank, args.global_rank,
                   torch.distributed.get_world_size(),
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

    # 手动设置全局变量
    mnt_dir = "/home/codereview"

    part = 6

    MASTER_HOST = "localhost"
    MASTER_PORT = 23333 + part - 1
    RANK = 0
    PER_NODE_GPU = 1
    WORLD_SIZE = 1
    NODES = 1
    NCCL_DEBUG = "INFO"

    # 手动设置args
    args.nproc_per_node = PER_NODE_GPU
    args.node_rank = RANK
    args.nnodes = NODES
    args.master_addr = MASTER_HOST
    args.master_port = MASTER_PORT

    # 服务器配置
    args.file_name = "checkpoints-57600-"
    args.model_name_or_path = '/data/lyf/code/Code_Reviewer/3_Pretrained_Model'
    args.model_name_or_path = '/data/lyf/code/Code_Reviewer/0_Result/2_only_explain_model'
    args.output_dir = '/data/lyf/code/Code_Reviewer/0_Result'
    args.load_model_path = '/data/lyf/code/Code_Reviewer/3_Pretrained_Model'
    args.load_model_path = '/data/lyf/code/Code_Reviewer/0_Result/2_only_explain_model'
    # args.eval_file = '/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/train_part_{}.jsonl'.format(part)
    args.eval_file = '/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/msg-test-focus-label.jsonl'
    args.exp_time = time.strftime("-%Y%m%d-%H_%M_%S", time.localtime(int(round(time.time() * 1000)) / 1000))

    args.focus_len = 10
    args.has_focus = False
    args.focus_file_name = 'focus_train_part_{}.txt'.format(part)
    args.focus_file_name = 'msg-valid.txt'
    args.do_test = True

    args.topk = 10
    args.preds_file_name = 'preds_{}.txt'.format(args.topk)

    # # 本地配置
    # args.model_name_or_path = r'E:\0_Code\postgraduate\CodeReviewer\3_Pretrained_Model'
    # args.output_dir = r'E:\0_Code\postgraduate\CodeReviewer\0_Result'
    # args.load_model_path = r'E:\0_Code\postgraduate\CodeReviewer\3_Pretrained_Model'
    # args.eval_file = r"E:\0_Code\postgraduate\CodeReviewer\2_Dataset\Comment_Generation\msg-test-small.jsonl"

    args.max_source_length = 512
    args.max_target_length = 128
    args.eval_batch_size = 6
    args.mask_rate = 0.15
    args.save_steps = 1800
    args.beam_size = 10
    args.long_steps = 100
    args.train_steps = 120000
    args.gpu_per_node = PER_NODE_GPU
    args.node_index = RANK
    args.seed = 2233
    args.raw_input = True

    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(WORLD_SIZE)
    os.environ['MASTER_ADDR'] = MASTER_HOST
    os.environ['MASTER_PORT'] = str(MASTER_PORT)
    os.environ['NCCL_DEBUG'] = NCCL_DEBUG

    # remove long tokenization warning. ref: https://github.com/huggingface/transformers/issues/991
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logger.info(args)
    print("Part: ", part)
    main(args)
    logger.info("Test finished.")
    # torch.multiprocessing.spawn(main, args=(args,), nprocs=torch.cuda.device_count())
