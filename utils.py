import csv
import re, json
import os, random
import torch, logging
from copy import deepcopy as cp
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import T5Tokenizer, RobertaTokenizer
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from matplotlib import pyplot as plt
import seaborn as sns
import nltk
from casual import seq_predict, load_model_and_tokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class MyTokenizer(object):
    """
    Wrapper for ByteLevelBPETokenizer
    """

    def __init__(self, vocab=None, merges=None, **kwargs):
        self.tokenizer = ByteLevelBPETokenizer(vocab, merges, **kwargs)
        self.update_id2token()

    @staticmethod
    def from_pretrained(path):
        vocabp = os.path.join(path, "vocab.json")
        mergesp = os.path.join(path, "merges.txt")
        mytoken = MyTokenizer(vocabp, mergesp)
        return mytoken

    def update_id2token(self):
        vocab = self.tokenizer.get_vocab()
        self.id2token = {vocab[token]: token for token in vocab}

    def add_special_tokens(self, dic):
        for values in dic.values():
            self.tokenizer.add_special_tokens(values)
        self.update_id2token()

    def convert_ids_to_tokens(self, ids):
        vocab = self.id2token
        return [vocab[i] for i in ids]

    def decode(self, ids, **kwargs):  ##### to be update
        tokens = self.convert_ids_to_tokens(ids)
        return " ".join(tokens)

    def encode(self, text, **kwargs):
        text = text.encode("ascii", errors="ignore").decode("ascii")
        return self.tokenizer.encode(text).ids

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def __len__(self):
        return len(self.tokenizer.get_vocab())


class RefineFeatures(object):
    def __init__(self, example_id, source_ids, target_ids):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids


class RefineDataset(Dataset):
    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1):
        self.tokenizer = tokenizer
        self.args = args
        logger.info("Reading examples from {}".format(file_path))
        examples = [json.loads(line) for line in open(file_path)]
        for i in range(len(examples)):
            if "id" not in examples[i]:
                examples[i]["id"] = i
        if samplenum > 0:
            examples = examples[:samplenum]
        logger.info(f"Tokenize examples: {file_path}")
        self.feats = pool.map(self.tokenize, \
                              [(example, tokenizer, args) for example in examples])

    def tokenize(self, item):
        example, tokenizer, args = item
        oldlines = example["old"].split("\n")
        newlines = example["new"].split("\n")
        oldlines = [line[1:].strip() for line in oldlines]
        newlines = [line[1:].strip() for line in newlines]
        oldlines = "\n".join(oldlines)
        newlines = "\n".join(newlines)
        oldlines = "<add>" + oldlines.replace("\n", "<add>")
        newlines = "<add>" + newlines.replace("\n", "<add>")
        comment = example["comment"]
        srcids = self.encode_remove(tokenizer, oldlines, args)
        srcids += [tokenizer.msg_id] + self.encode_remove(tokenizer, comment, args)
        tgtids = self.encode_remove(tokenizer, newlines, args)
        srcids, tgtids = self.pad_assert(srcids, tgtids, args, tokenizer)
        return RefineFeatures(example["id"], srcids, tgtids)

    @staticmethod
    def process_pred_gold(pred, gold):
        gold = gold.split("\n")
        gold = [line[1:].strip() for line in gold]
        gold = " ".join(gold)
        pred = " ".join(pred.split())
        pred = pred.replace("<add> ", "")
        return pred, gold

    def pad_assert(self, source_ids, target_ids, args, tokenizer):
        source_ids = source_ids[:args.max_source_length - 2]
        source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
        pad_len = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_id] * pad_len
        target_ids = target_ids[:args.max_target_length - 2]
        target_ids = [tokenizer.bos_id] + target_ids + [tokenizer.eos_id]
        pad_len = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_id] * pad_len
        assert len(source_ids) == args.max_source_length, "Not equal length."
        assert len(target_ids) == args.max_target_length, "Not equal length."
        return source_ids, target_ids

    def encode_remove(self, tokenizer, text, args):
        text = tokenizer.encode(text, max_length=args.max_source_length, truncation=True)
        if type(tokenizer) == T5Tokenizer:
            return text[:-1]
        elif type(tokenizer) == RobertaTokenizer:
            return text[1:-1]
        elif type(tokenizer) == MyTokenizer:
            return text
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]


class SimpleRefineDataset(RefineDataset):
    def tokenize(self, item):
        example, tokenizer, args = item
        oldlines = example["old"].split("\n")
        newlines = example["new"].split("\n")
        oldlines = [line[1:].strip() for line in oldlines]
        newlines = [line[1:].strip() for line in newlines]
        oldlines = " ".join(oldlines)
        newlines = " ".join(newlines)
        comment = example["comment"]
        srcids = self.encode_remove(tokenizer, oldlines, args)
        srcids += [tokenizer.msg_id] + self.encode_remove(tokenizer, comment, args)
        tgtids = self.encode_remove(tokenizer, newlines, args)
        srcids, tgtids = self.pad_assert(srcids, tgtids, args, tokenizer)
        return RefineFeatures(example["id"], srcids, tgtids)

    @staticmethod
    def process_pred_gold(pred, gold):
        gold = gold.split("\n")
        gold = [line[1:].strip() for line in gold]
        gold = " ".join(gold)
        pred = " ".join(pred.split())
        return pred, gold


class Seq2SeqDataset(RefineDataset):
    def tokenize(self, item):
        example, tokenizer, args = item
        inputs, outputs = example["old"], example["new"]
        inputs = " ".join(inputs.split())
        outputs = " ".join(outputs.split())
        srcids = self.encode_remove(tokenizer, inputs, args)
        tgtids = self.encode_remove(tokenizer, outputs, args)
        srcids, tgtids = self.pad_assert(srcids, tgtids, args, tokenizer)
        return RefineFeatures(example["id"], srcids, tgtids)

    @staticmethod
    def process_pred_gold(pred, gold):
        gold = " ".join(gold.split())
        pred = " ".join(pred.split())
        return pred, gold


class TextDataset(Dataset):
    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1):
        self.cnt = 0
        self.tokenizer = tokenizer
        self.args = args
        if isinstance(tokenizer, MyTokenizer):
            tokenizer_type = "mytok"
        elif isinstance(tokenizer, T5Tokenizer):
            tokenizer_type = ""
        elif isinstance(tokenizer, RobertaTokenizer):
            tokenizer_type = "rb"
        else:
            tokenizer_type = "unk"
        savep = file_path.replace(".jsonl", tokenizer_type + ".exps")
        # savep = "/home/v-zhuoli1/lzzz/processed/chunk_25.exps"
        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))
            examples = torch.load(savep)
        else:
            logger.info("Reading examples from {}".format(file_path))
            examples = read_review_examples(file_path, samplenum, tokenizer)
            logger.info(f"Tokenize examples: {file_path}")
            examples = pool.map(self.tokenize, \
                                [(example, tokenizer, args) for example in examples])
            torch.save(examples, savep)
        logger.info("Convert examples to features...")
        self.set_start_end_ids(examples)
        self.featss = pool.map(self.convert_examples_to_features, \
                               [(example, tokenizer, args) for example in examples])
        self.feats = [feat for feats in self.featss for feat in feats]  # expand the lists

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i]

    def reset_len(self, data_len):
        assert len(self.feats) >= data_len
        self.feats = self.feats[:data_len]

    def set_start_end_ids(self, examples):
        for example in examples:
            labels = example.labels
            start_id = 0
            end_id = len(labels) - 1
            for i, label in enumerate(labels):
                if label != -100:  # find the first label
                    start_id = i
                    break
            for i in range(len(labels) - 1, -1, -1):
                label = labels[i]
                if label != -100:
                    end_id = i
                    break
            example.start_id = start_id
            example.end_id = end_id

    def tokenize(self, item):
        example, tokenizer, args = item
        example.input = self.encode_remove(tokenizer, example.input, args)
        e0id = tokenizer.special_dict["<e0>"]
        inputs = " ".join(str(id) for id in example.input)
        lines = inputs.split(" " + str(e0id) + " ")
        lines = [
            [int(v) for v in line.split(" ") if len(v) > 0] for line in lines
        ]
        lens = [len(line) for line in lines]
        # if 0 in lens:
        #     logger.info("Warning: empty line in an example.")
        lens = list(map(len, lines))
        curlen = len(lens) + sum(lens)
        left, right = 0, len(lines)
        while curlen > args.max_source_length - 2:
            if left % 2 == 0:
                curlen -= 1 + len(lines[left])
                left += 1
            else:
                right -= 1
                curlen -= 1 + len(lines[right])
        lines = lines[left:right]
        labels = example.labels[left:right]
        assert len(lines) + sum(
            map(len, lines)) <= args.max_source_length - 2, "Too long inputs in TextDataset.tokenize."
        if len(lines) != len(labels):
            logger.info("Not equal length in TextDataset.tokenize.")
            lines = lines[:len(labels)]
            labels = labels[:len(lines)]
        example.lines = lines
        example.labels = labels
        example.msg = self.encode_remove(tokenizer, example.msg, args)
        return example

    def convert_examples_to_features(self, item):
        example, _, _ = item
        if len(example.msg) > 0:
            exs = []
            for _ in range(3):  # up sampling
                if random.random() < 0.5:
                    exs.append(self.genmsg_example(item))
                else:
                    exs.append(self.daemsg_example(item))
            return exs
        if random.random() < 0.5:
            return [self.encoder_example(item)]
        return [self.decoder_example(item)]

    def encoder_example(self, item):
        example, tokenizer, args = item
        lines = example.lines
        labels = example.labels
        target_ids = [tokenizer.pad_id] * args.max_target_length
        source_ids, input_labels = [], []
        for i, (line, label) in enumerate(zip(lines, labels)):
            if i == example.start_id:
                source_ids.append(tokenizer.start_id)
                input_labels.append(-100)
            if label != -100:  # only insert special tokens at diffs, not context
                source_ids.append(tokenizer.mask_id)
                input_labels.append(label)
            source_ids.extend(line)
            input_labels.extend([-100] * len(line))
            if i == example.end_id:
                source_ids.append(tokenizer.end_id)
                input_labels.append(-100)
        assert len(input_labels) == len(source_ids), "Not equal length."
        assert len(input_labels) <= args.max_source_length, f"Too long inputs: {len(input_labels)}."
        source_ids = source_ids[:args.max_source_length - 2]
        input_labels = input_labels[:args.max_source_length - 2]
        source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
        input_labels = [-100] + input_labels + [-100]
        pad_len = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_id] * pad_len
        input_labels += [-100] * pad_len

        new_input_labels = []
        map_dict = {0: tokenizer.del_id, 1: tokenizer.add_id, 2: tokenizer.keep_id}
        for label in input_labels:
            if label == -100:
                new_input_labels.append(-100)
            else:
                new_input_labels.append(map_dict[label])
        input_labels = new_input_labels
        assert len(source_ids) == args.max_source_length, "Not equal length."
        assert len(input_labels) == args.max_source_length, "Not equal length."
        return ReviewFeatures(example.idx, source_ids, input_labels, target_ids, type="label")

    def decoder_example(self, item):
        example, tokenizer, args = item
        lines = example.lines
        labels = example.labels

        input_labels = [-100] * args.max_source_length
        source_ids, target_ids = [], []
        SPECIAL_ID = 0
        mask_idxs = random.choices(range(len(lines)), k=int(len(lines) * args.mask_rate))
        id_dict = {0: tokenizer.del_id, 1: tokenizer.add_id, 2: tokenizer.keep_id}
        for i, (line, label) in enumerate(zip(lines, labels)):
            if i == example.start_id:
                source_ids.append(tokenizer.start_id)
            if label in id_dict:
                source_ids.append(id_dict[label])
            if i in mask_idxs:
                source_ids.append(tokenizer.special_dict[f"<e{SPECIAL_ID}>"])
                target_ids.append(tokenizer.special_dict[f"<e{SPECIAL_ID}>"])
                target_ids.extend(line)
                if SPECIAL_ID < 99:  # only 0-99 ids in vocab
                    SPECIAL_ID += 1
            else:
                source_ids.extend(line)
            if i == example.end_id:
                source_ids.append(tokenizer.end_id)
        source_ids, target_ids = self.pad_assert(source_ids, target_ids, args, tokenizer)
        return ReviewFeatures(example.idx, source_ids, input_labels, target_ids, type="line")

    def genmsg_example(self, item):
        example, tokenizer, args = item
        lines = example.lines
        labels = example.labels
        input_labels = [-100] * args.max_source_length
        source_ids, target_ids = [], []
        id_dict = {0: tokenizer.del_id, 1: tokenizer.add_id, 2: tokenizer.keep_id}
        for i, (line, label) in enumerate(zip(lines, labels)):
            if i == example.start_id:
                source_ids.append(tokenizer.start_id)
            if label != -100:
                source_ids.append(id_dict[label])
            source_ids.extend(line)
            if i == example.end_id:
                source_ids.append(tokenizer.end_id)
        target_ids.append(tokenizer.msg_id)
        target_ids.extend(example.msg)
        assert len(source_ids) <= args.max_source_length, f"Too long inputs: {len(source_ids)}."
        source_ids, target_ids = self.pad_assert(source_ids, target_ids, args, tokenizer)
        return ReviewFeatures(example.idx, source_ids, input_labels, target_ids, type="genmsg")

    def daemsg_example(self, item):
        example, tokenizer, args = item
        input_labels = [-100] * args.max_source_length
        source_ids, target_ids = [], []
        msg_ids = cp(example.msg)
        masks = [random.random() < 0.20 for _ in range(len(msg_ids))]
        if sum(masks) == 0:
            idx = random.choice(range(len(msg_ids)))
            masks[idx] = True
        source_ids, target_ids = [], []
        i = 0
        SPECIAL_ID = 0
        while i < len(masks):
            j = i
            while j < len(masks) and not masks[j]:
                source_ids.append(msg_ids[j])
                j += 1
            if j == len(masks):
                break
            source_ids.append(tokenizer.special_dict[f"<e{SPECIAL_ID}>"])
            target_ids.append(tokenizer.special_dict[f"<e{SPECIAL_ID}>"])
            while j < len(masks) and masks[j]:
                target_ids.append(msg_ids[j])
                j += 1
            if SPECIAL_ID < 99:  # only 0-99 ids in vocab
                SPECIAL_ID += 1
            i = j
        source_ids, target_ids = self.pad_assert(source_ids, target_ids, args, tokenizer)
        return ReviewFeatures(example.idx, source_ids, input_labels, target_ids, type="daemsg")

    def pad_assert(self, source_ids, target_ids, args, tokenizer):
        source_ids = source_ids[:args.max_source_length - 2]
        source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
        pad_len = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_id] * pad_len
        target_ids = target_ids[:args.max_target_length - 1]
        target_ids = target_ids + [tokenizer.eos_id]
        pad_len = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_id] * pad_len
        assert len(source_ids) == args.max_source_length, "Not equal length."
        assert len(target_ids) == args.max_target_length, "Not equal length."
        return source_ids, target_ids

    def encode_remove(self, tokenizer, text, args):
        text = tokenizer.encode(text, max_length=args.max_source_length, truncation=True)
        if type(tokenizer) == T5Tokenizer:
            return text[:-1]
        elif type(tokenizer) == RobertaTokenizer:
            return text[1:-1]
        elif type(tokenizer) == MyTokenizer:
            return text
        else:
            raise NotImplementedError


class CommentGenDataset(TextDataset):
    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1):
        # 初始化 tokenizer，并根据 tokenizer 的类型设置 tokenizer_type
        self.tokenizer = tokenizer
        if isinstance(tokenizer, MyTokenizer):
            tokenizer_type = "mytok"  # 自定义的 Tokenizer
        elif isinstance(tokenizer, T5Tokenizer):
            tokenizer_type = ""  # T5 Tokenizer
        elif isinstance(tokenizer, RobertaTokenizer):
            tokenizer_type = "rb"  # Roberta Tokenizer
        else:
            tokenizer_type = "unk"  # 未知 Tokenizer

        # 替换文件扩展名以创建已处理示例的保存路径。
        savep = file_path.replace(".jsonl", tokenizer_type + ".exps")

        # 如果预处理后的特征文件存在，则从文件加载，否则从原始数据处理。
        if os.path.exists(savep):
            logger.info("从 {} 加载示例".format(savep))  # 从文件加载示例
            examples = torch.load(savep)
        else:
            logger.info("从 {} 读取示例".format(file_path))  # 读取原始文件中的数据
            examples = read_review_examples(file_path, samplenum, tokenizer)
            # 可选的对评论进行词语分割（示例中此行已注释）
            # for i in range(len(examples)):
            #     examples[i].msg = " ".join(nltk.word_tokenize(examples[i].msg))
            logger.info("对示例进行分词处理: {}".format(file_path))
            # 使用进程池来加速示例的分词处理
            examples = pool.map(self.tokenize, \
                                [(example, tokenizer, args) for example in examples])
            torch.save(examples, savep)  # 保存处理后的数据

        logger.info("将示例转换为特征...")
        # 设置每个示例的起始和结束标识符
        self.set_start_end_ids(examples)
        # 使用进程池转换所有示例为特征
        self.feats = pool.map(self.convert_examples_to_features, \
                              [(example, tokenizer, args) for example in examples])
        # 过滤掉可能转换失败返回 None 的特征
        self.feats = [feat for feat in self.feats if feat is not None]

    def convert_examples_to_features(self, item):
        # 解包输入元组
        example, tokenizer, args = item
        # 如果评论信息为空，则不生成特征
        if len(example.msg) == 0:
            return None
        # 生成评论特征
        return self.genmsg_example(item)


class CommentClsDataset(TextDataset):
    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1):
        self.tokenizer = tokenizer
        if isinstance(tokenizer, MyTokenizer):
            tokenizer_type = "mytok"
        elif isinstance(tokenizer, T5Tokenizer):
            tokenizer_type = ""
        elif isinstance(tokenizer, RobertaTokenizer):
            tokenizer_type = "rb"
        else:
            tokenizer_type = "unk"
        savep = file_path.replace(".jsonl", tokenizer_type + ".exps")
        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))
            examples = torch.load(savep)
        else:
            logger.info("Reading examples from {}".format(file_path))
            examples = read_review_examples(file_path, samplenum, tokenizer)
            logger.info(f"Tokenize examples: {file_path}")
            examples = pool.map(self.tokenize, \
                                [(example, tokenizer, args) for example in examples])
            torch.save(examples, savep)
        logger.info("Convert examples to features...")
        self.set_start_end_ids(examples)
        self.feats = pool.map(self.convert_examples_to_features, \
                              [(example, tokenizer, args) for example in examples])

    def convert_examples_to_features(self, item):
        example, tokenizer, args = item
        tmpfeature = self.genmsg_example(item)
        return ClsFeatures(tmpfeature.example_id, tmpfeature.source_ids, example.y)


class SimpleClsDataset(TextDataset):
    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1):
        self.tokenizer = tokenizer
        if isinstance(tokenizer, MyTokenizer):
            tokenizer_type = "mytok"
        elif isinstance(tokenizer, T5Tokenizer):
            tokenizer_type = ""
        elif isinstance(tokenizer, RobertaTokenizer):
            tokenizer_type = "rb"
        else:
            tokenizer_type = "unk"
        savep = file_path.replace(".jsonl", tokenizer_type + ".simpexps")
        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))
            self.feats = torch.load(savep)
        else:
            logger.info("Reading examples from {}".format(file_path))
            examples = read_review_examples(file_path, samplenum, tokenizer)
            logger.info(f"Tokenize examples: {file_path}")
            self.feats = pool.map(self.convert_examples_to_features, \
                                  [(example, tokenizer, args) for example in examples])
            torch.save(self.feats, savep)

    def convert_examples_to_features(self, item):
        example, tokenizer, args = item
        example.input_lines = example.input.split("<e0>")
        labels_l = len(example.labels)
        example.input_lines = example.input_lines[:labels_l]
        for i in range(len(example.input_lines)):
            if example.labels[i] == 1:
                example.input_lines[i] = "+ " + example.input_lines[i]
            elif example.labels[i] == 0:
                example.input_lines[i] = "- " + example.input_lines[i]
        example.input = " ".join(example.input_lines)
        input_ids = self.encode_remove(tokenizer, example.input, args)
        exceed_l = len(input_ids) - args.max_source_length + 2
        if exceed_l > 0:
            halfexl = (exceed_l + 1) // 2
            input_ids = input_ids[halfexl:-halfexl]
        source_ids = input_ids[:args.max_source_length - 2]
        source_ids = [tokenizer.bos_id] + source_ids + [tokenizer.eos_id]
        pad_len = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_id] * pad_len
        example_id = example.idx
        y = example.y
        return ClsFeatures(example_id, source_ids, y)


class SimpleGenDataset(TextDataset):
    def __init__(self, tokenizer, pool, args, file_path, samplenum=-1):
        # 初始化 tokenizer，并根据 tokenizer 的类型设置 tokenizer_type
        self.tokenizer = tokenizer
        if isinstance(tokenizer, MyTokenizer):
            tokenizer_type = "mytok"  # 自定义的 Tokenizer
        elif isinstance(tokenizer, T5Tokenizer):
            tokenizer_type = ""  # T5 Tokenizer
        elif isinstance(tokenizer, RobertaTokenizer):
            tokenizer_type = "rb"  # Roberta Tokenizer
        else:
            tokenizer_type = "unk"  # 未知 Tokenizer

        # 替换文件扩展名以创建已处理示例的保存路径。
        savep = file_path.replace(".jsonl", tokenizer_type + ".simpgenexps")

        # 如果预处理后的特征文件存在，则从文件加载，否则从原始数据处理。
        if os.path.exists(savep):
            logger.info("Loading examples from {}".format(savep))  # 从文件加载示例
            self.feats = torch.load(savep)
        else:
            logger.info("Reading examples from {}".format(file_path))  # 读取原始文件中的数据
            data = read_jsonl(file_path)
            # data = [dic for dic in data if len(dic["patch"].split("\n")) <= 20]
            # 为每个数据项分配一个索引以跟踪。
            for i in range(len(data)):
                data[i]["idx"] = i

            logger.info("Tokenize examples: {}".format(file_path))  # 对示例进行分词处理
            # 转换所有示例为特征（这里展示的是单线程版本）
            self.feats = [self.convert_examples_to_features((dic, tokenizer, args)) for dic in data]
            torch.save(self.feats, savep)  # 保存处理后的特征数据

    def convert_examples_to_features(self, item):
        # 解包输入元组
        dic, tokenizer, args = item
        # 从字典中获取代码差异和消息
        diff, msg = dic["patch"], dic["msg"]
        # 处理 diff，移除初始行和任何空行
        difflines = diff.split("\n")[1:]
        difflines = [line for line in difflines if line.strip()]  # 去掉空行

        has_explain = dic.get("has_explain", 0)

        # 映射每行 diff 的第一个字符到数字标签
        map_dic = {"-": 0, "+": 1, " ": 2}  # 映射 diff 符号到数字

        def f(s):
            return map_dic.get(s, 2)  # 将 diff 行首的字符转换为对应的数字标签

        # 为 diff 中的每一行生成标签
        labels = [f(line[0]) for line in difflines]
        # 去除行首的符号，并去掉空白字符
        difflines = [line[1:].strip() for line in difflines]

        # 根据标签构建输入字符串
        inputstr = ""

        # 添加注意力标识
        if args.has_focus:
            if "true_focus" in dic and "other_focus" in dic:
                true_focus = dic["true_focus"]
                inputstr += "<true_focus>" + " ".join(true_focus)
                other_focus = dic["other_focus"]
                inputstr += "<other_focus>" + " ".join(other_focus)
            else:
                raise ValueError("Focus not found in the data.")

        for label, line in zip(labels, difflines):
            if label == 1:
                inputstr += "<add>" + line
            elif label == 0:
                inputstr += "<del>" + line
            else:
                inputstr += "<keep>" + line

        # 对输入字符串进行编码
        source_ids = self.encode_remove(tokenizer, inputstr, args)
        # 初始化目标 ID 列表
        target_ids = [tokenizer.msg_id]  # 加入特殊的消息标识符
        # 编码消息
        msg = self.encode_remove(tokenizer, dic["msg"], args)
        target_ids.extend(msg)
        # 确保 source_ids 和 target_ids 的长度一致
        source_ids, target_ids = self.pad_assert(source_ids, target_ids, args, tokenizer)
        # 初始化输入标签，用于后续模型训练
        input_labels = [-100] * len(source_ids)

        # 返回处理好的特征
        return ReviewFeatures(dic["idx"], source_ids, input_labels, target_ids, "genmsg", has_explain)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self, example_id, source_ids, target_ids, url=None):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class ReviewFeatures(object):
    def __init__(self, example_id, source_ids, source_labels, target_ids, type, has_explain):
        self.example_id = example_id
        self.source_ids = source_ids
        self.source_labels = source_labels
        self.target_ids = target_ids
        assert type in ("label", "line", "genmsg", "daemsg")
        self.type = type
        self.has_explain = has_explain


class ClsFeatures(object):
    def __init__(self, example_id, source_ids, y):
        self.example_id = example_id
        self.source_ids = source_ids
        self.y = y


class ReviewExample(object):
    """A single training/test example."""

    def __init__(
            self, idx, oldf, diff, msg, cmtid, max_len, y
    ):
        self.idx = idx  # idx is useless yet
        self.oldf = oldf
        self.diff = diff
        self.msg = msg
        self.cmtid = cmtid
        self.max_len = max_len
        self.y = y
        self.prevlines = []
        self.afterlines = []
        self.lines = []
        self.labels = []
        self.avail = False
        self.input = ""
        self.align_and_clean()
        self.postprocess()

    def postprocess(self):
        if not self.avail:
            return
        # Warning: lines is not self.lines
        # lines for rough length estimation
        lines = [source_str.split() for source_str in self.lines]
        inputl = len(lines)  # line tag
        inputl += sum(map(len, lines))
        left, right = 0, len(lines)
        while inputl > self.max_len:
            if left % 2 == 0:
                inputl -= len(lines[left]) + 1
                left += 1
            else:
                right -= 1
                inputl -= len(lines[right]) + 1
        lines = lines[left:right]
        self.lines = self.lines[left:right]
        self.labels = self.labels[left:right]
        prevlines = self.prevlines
        afterlines = self.afterlines
        prev_after_len = max(len(prevlines), len(afterlines))
        i = 0
        while inputl < self.max_len and i < prev_after_len:
            if i < len(prevlines):
                newl = inputl + len(prevlines[-1 - i].split()) + 1
                if newl > self.max_len:
                    break
                self.lines.insert(0, prevlines[-1 - i])
                self.labels.insert(0, -100)
                inputl = newl  # tag
            if i < len(afterlines):
                newl = inputl + len(afterlines[i].split()) + 1
                if newl > self.max_len:
                    break
                self.lines.append(afterlines[i])
                self.labels.append(-100)
                inputl = newl  # tag
            i += 1
        assert inputl <= self.max_len, "Too long inputs."
        assert len(self.lines) == len(self.labels), "Not equal length."
        self.input = "<e0>".join(self.lines)
        self.prevlines, self.lines, self.afterlines = [], [], []

    def remove_space_clean(self, line):
        """
            Remove start and end empty chars.
        """
        rep = " \t\r"
        totallen = len(line)
        i = 0
        while i < totallen and line[i] in rep:
            i += 1
        j = totallen - 1
        while j >= 0 and line[j] in rep:
            j -= 1
        line = line[i: j + 1]
        return line

    def align_and_clean(self):
        oldflines = self.oldf.split("\n")
        difflines = self.diff.split("\n")
        first_line = difflines[0]
        difflines = difflines[1:]
        difflines = [line for line in difflines if line != r"\ No newline at end of file"]
        regex = r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@"
        matchres = re.match(regex, first_line)
        if matchres:
            startline, rangelen, startpos, endpos = matchres.groups()
            self.avail = True
        else:
            self.avail = False
            return
        startline, rangelen = int(startline) - 1, int(rangelen)
        endline = startline + rangelen
        self.prevlines = oldflines[:startline]
        self.afterlines = oldflines[endline:]
        for line in difflines:
            if line.startswith("-"):
                self.lines.append(line[1:])
                self.labels.append(0)
            elif line.startswith("+"):
                self.lines.append(line[1:])
                self.labels.append(1)
            else:
                self.lines.append(line)
                self.labels.append(2)
        self.prevlines = [self.remove_space_clean(line) for line in self.prevlines]
        self.afterlines = [self.remove_space_clean(line) for line in self.afterlines]
        self.lines = [self.remove_space_clean(line) for line in self.lines]
        self.msg = self.remove_space_clean(self.msg)
        self.prevlines = [line for line in self.prevlines if len(line) > 0]
        self.afterlines = [line for line in self.afterlines if len(line) > 0]
        # print("\n".join(self.prevlines))
        # print("\n\n\n\n")
        # print("\n".join(self.lines))
        # print("\n\n\n\n")
        # print("\n".join(self.afterlines))
        # print("\n\n\n\n")
        assert len(self.lines) == len(self.labels), "Not equal length in align."
        topack = list(
            zip(
                *[
                    (line, label)
                    for line, label in zip(self.lines, self.labels)
                    if len(line) > 0
                ]
            )
        )
        if topack == []:
            self.avail = False
            return
        else:
            self.lines, self.labels = topack
        # tuple->list, convenient for later operation
        self.lines = list(self.lines)
        self.labels = list(self.labels)


def read_review_examples(filename, data_num=-1, tokenizer=None):
    """Read examples from filename."""
    examples = []
    idx = 0
    with open(filename) as f:
        for line in f:
            try:
                js = json.loads(line.strip())
            except:
                print("Error during reading json data.")
                continue
            maxl = 200
            if "y" not in js:
                js["y"] = 0
            if "msg" in js and len(js["msg"]) > 0:
                js["y"] = 1
            example = ReviewExample(
                idx=idx,
                oldf=js["oldf"],
                diff=js["patch"],
                msg=js["msg"] if "msg" in js else "",
                cmtid=js["cmtid"] if "cmtid" in js else "",
                max_len=maxl,
                y=js["y"]
            )
            if example.avail:
                examples.append(example)
                idx += 1
                if idx == data_num:
                    break
            else:
                # print(f"Passing {idx} because of invalid diff.")
                idx += 1
                if idx == data_num:
                    break

    return examples


def read_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            try:
                js = json.loads(line.strip())
            except:
                print("Error during reading json data.")
                continue
            data.append(js)
    return data


# 打印
def attention_plot(attention, x_texts, y_texts=None, figsize=(15, 10), annot=False, figure_path='./figures',
                   figure_name='attention_weight.png'):
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(attention,
                     cbar=True,
                     cmap="RdBu_r",
                     annot=annot,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 10},
                     yticklabels=y_texts,
                     xticklabels=x_texts
                     )
    if os.path.exists(figure_path) is False:
        os.makedirs(figure_path)
    plt.savefig(os.path.join(figure_path, figure_name))
    plt.show()
    plt.close()


def read_stopwords(file_path):
    stopwords = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 去除行末的换行符
            line = line.strip()
            # 跳过空行
            if not line:
                continue
            words = line.split(' ')
            for word in words:
                stopwords.append(word)
    return stopwords


def filter_stopwords(tokens_dict):
    filter_tokens = {}
    # 调用函数读取停用词列表
    stopwords_list = read_stopwords(os.path.join(os.path.dirname(__file__), "stop_words.txt"))
    # 正确地遍历字典的键和值
    for token, value in tokens_dict.items():
        # 假设单词不是停用词
        is_stopword = any(stopword in token for stopword in stopwords_list)
        # 如果不是停用词，添加到新字典中
        if not is_stopword:
            filter_tokens[token] = value
    return filter_tokens


def top_k_token_dict(tokens, values, k):
    # 创建字典
    token_dict = {}
    for i in range(len(tokens)):
        token_dict[tokens[i]] = values[i]
    # 过滤停用词
    filter_dict = filter_stopwords(token_dict)

    # 直接使用字典切片来获取前k个键值对
    return dict(list(filter_dict.items())[:k])


def merge_dict(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            dict1[key] += value
        else:
            dict1[key] = value

    return dict1


def read_focus(path):
    data = []
    with open(path, "r") as file:
        for line in file.readlines():
            # 使用 strip() 方法去除每个值的前后空格，并将结果以逗号分隔的形式拆分为列表
            row = [value.strip() for value in line.strip().split(',')]
            data.append(row)
    return data


def add_focus_info(json_path, focus_path, new_file_path):
    json_data = read_jsonl(json_path)
    focus_data = read_focus(focus_path)
    for idx, data in enumerate(json_data):
        print(f"Processing {idx}...")
        data['focus'] = focus_data[idx]

    with open(new_file_path, 'w') as f:
        print(f"Writing to {new_file_path}...")
        for obj in json_data:
            f.write(json.dumps(obj) + '\n')


def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')


def split_jsonl(json_path, output_dir, num_files=5):
    json_data = read_jsonl(json_path)
    total_lines = len(json_data)
    lines_per_file = (total_lines + num_files - 1) // num_files  # Ensure even distribution

    for i in range(num_files):
        start_index = i * lines_per_file
        end_index = min(start_index + lines_per_file, total_lines)
        chunk = json_data[start_index:end_index]
        new_file_path = os.path.join(output_dir, f'train_part_{i + 1}.jsonl')
        write_jsonl(chunk, new_file_path)


def merge_jsonl(json_path_list, output_file):
    merged_data = []
    for json_path in json_path_list:
        merged_data.extend(read_jsonl(json_path))
    write_jsonl(merged_data, output_file)


# 遍历jsonl文件，找到msg属性中有 because 的数据
def find_because(json_path):
    json_data = read_jsonl(json_path)
    because_data = []
    for idx, data in enumerate(json_data):
        if "because" in data["msg"]:
            because_data.append(data['msg'])
    return because_data


def get_all_msg(json_path):
    json_data = read_jsonl(json_path)
    all_msg = []
    for data in json_data:
        all_msg.append(data['msg'])
        print(data['msg'])
    return all_msg


def save_to_csv(data, csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text'])  # 写入表头
        for msg in data:
            writer.writerow([msg])


def add_target_focus(json_path, output_path):
    # 用于匹配反引号包围的内容的正则表达式
    pattern = re.compile(r'`(.*?)`')
    data = read_jsonl(json_path)
    for line in data:
        msg = line['msg']
        matches = pattern.findall(msg)
        tfocus = ' '.join(matches)
        focus = line['focus']
        true_focus = []
        other_focus = []
        for f in focus:
            if f in msg:
                true_focus.append(f)
            else:
                other_focus.append(f)

        line['true_focus'] = true_focus
        line['other_focus'] = other_focus

    with open(output_path, 'w') as f:
        print(f"Writing to {output_path}...")
        for obj in data:
            f.write(json.dumps(obj) + '\n')


def read_explain(file_path):
    explains = {}
    with open(file_path, 'r') as f:
        # 跳过第一行
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                key = parts[0]
                seq_pred = int(parts[1])
                explains[key] = seq_pred
    return explains


def add_explain_info(json_path, explain_path, new_file_path):
    json_data = read_jsonl(json_path)
    explain_data = read_explain(explain_path)

    for idx, data in enumerate(json_data):
        key = f"dcorpus_ddocid_{idx}_0"
        data['has_explain'] = explain_data.get(key, None)

    with open(new_file_path, 'w') as f:
        print(f"Writing to {new_file_path}...")
        for obj in json_data:
            f.write(json.dumps(obj) + '\n')


def calculate_casual_percent(model_name_or_path, file_path, cache_dir=None):
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, cache_dir)
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
    local_rank = 1
    logits = seq_predict(model, tokenizer, lines, local_rank)
    # 对logits进行argmax操作
    predictions = torch.argmax(logits, dim=1)

    # 统计标签1的个数
    count_label_1 = (predictions == 1).sum().item()

    # 计算标签1的百分比
    total_predictions = predictions.size(0)
    percent_label_1 = (count_label_1 / total_predictions) * 100

    return count_label_1, percent_label_1


if __name__ == '__main__':
    # 读取jsonl
    # file_path = r"E:\0_Code\postgraduate\CodeReviewer\2_Dataset\Comment_Generation\msg-train.jsonl"
    # data = read_jsonl(file_path)

    # 为jsonl添加focus信息
    # part = 2
    # json_file = r"/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/msg-test.jsonl"
    # focus_file = r"/data/lyf/code/Code_Reviewer/0_Result/msg-test.txt"
    # new_file = r"/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/msg-test-focus.jsonl"
    # add_focus_info(json_path=json_file, focus_path=focus_file, new_file_path=new_file)

    # 划分jsonl
    # json_file = r"/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/msg-train.jsonl"
    # output_dir = r"/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation"
    # split_jsonl(json_file, output_dir, num_files=5)

    # 合并jsonl
    # jsonl_list = [
    #     r"/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/new_train_part_1.jsonl",
    #     r"/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/new_train_part_2.jsonl",
    #     r"/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/new_train_part_3.jsonl",
    #     r"/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/new_train_part_4.jsonl",
    #     r"/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/new_train_part_5.jsonl"
    # ]
    # output_file = r"/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/msg-train-focus.jsonl"
    # merge_jsonl(jsonl_list, output_file)

    # 获取所有msg数据 存储为csv
    # jsonl_file = r'/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/msg-train.jsonl'
    # csv_file = r'/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/msg-train.csv'
    # all_msg = get_all_msg(json_path=jsonl_file)
    # save_to_csv(all_msg, csv_file)

    # 添加目标关注点信息
    # json_file = r'/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/msg-test-focus-label.jsonl'
    # output_file = r'/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/msg-test-focus-label.jsonl'
    # add_target_focus(json_file, output_file)

    # 添加has_explain信息
    # json_file = r'/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/msg-train-focus-label-explain.jsonl'
    # explain_file = r'/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/msg-train-explain.txt'
    # new_json_file = r'/data/lyf/code/Code_Reviewer/2_Dataset/Comment_Generation/msg-train-focus-label-explain.jsonl'
    #
    # json_file = r"E:\0_Code\postgraduate\CodeReviewer\2_Dataset\Comment_Generation\msg-train-explain.jsonl"
    # explain_file = r"E:\0_Code\postgraduate\CodeReviewer\2_Dataset\Comment_Generation\msg-train-explain.txt"
    # new_json_file = r"E:\0_Code\postgraduate\CodeReviewer\2_Dataset\Comment_Generation\msg-train-explain.jsonl"
    #
    # add_explain_info(json_file, explain_file, new_json_file)

    # 计算解释性信息占比
    casual_seq_model_path = "/data/lyf/code/Code_Reviewer/2_Dataset/seq-baseline"
    file_name = r'/data/lyf/code/Code_Reviewer/0_Result/1_preds_only_reinforce/preds_origin_topk1.txt'
    count, percent = calculate_casual_percent(casual_seq_model_path, file_name)
    print(f"标签1的个数: {count}")
    print(f"标签1的百分比: {percent:.2f}%")
