import json

import torch
import nltk
from altair.utils.execeval import eval_block
from rouge.rouge import Rouge

from casual import load_model_and_tokenizer, seq_predict
from evaluator.smooth_bleu import computeMaps, bleuFromMaps


def read_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]

    return lines


def tokenize(texts):
    processed_texts = []
    for text in texts:
        tokenized_text = nltk.wordpunct_tokenize(text)
        processed_text = " ".join(tokenized_text)
        processed_texts.append(processed_text)
    return processed_texts


def remove_stopwords(texts):
    with open(stopwords_path, 'r') as f:
        stopwords = [line.strip() for line in f]
    processed_texts = []
    for text in texts:
        filtered_words = [word for word in text.split() if word not in stopwords]
        processed_text = " ".join(filtered_words)
        processed_texts.append(processed_text)

    return processed_texts


def eval_bleu(predict_file, gold_file, rmstop):
    print('Start to evaluate the BLEU score...')
    predictions = tokenize(read_txt(predict_file))
    golds = tokenize(read_txt(gold_file))

    if rmstop:
        predictions = remove_stopwords(predictions)
        golds = remove_stopwords(golds)

    # 将预测结果和参考答案格式化为模型期望的形式
    predictions = [str(i) + "\t" + pred.replace("\t", " ") for (i, pred) in enumerate(predictions)]
    golds = [str(i) + "\t" + gold.replace("\t", " ") for (i, gold) in enumerate(golds)]

    # 计算预测结果与参考答案之间的映射关系
    goldMap, predictionMap = computeMaps(predictions, golds)

    # 计算BLEU值
    bleu = bleuFromMaps(goldMap, predictionMap)[0]
    print("BLEU ", 'w/o stopwords' if rmstop else 'w/ stopwords', ': ', bleu)


def eval_rouge(predict_file, gold_file, rmstop):
    print('Start to evaluate the ROUGE score...')
    predictions = tokenize(read_txt(predict_file))
    golds = tokenize(read_txt(gold_file))

    if rmstop:
        predictions = remove_stopwords(predictions)
        golds = remove_stopwords(golds)

    for i in range(len(predictions)):
        if predictions[i] == ' ':
            print('empty prediction: ', i)
        if golds[i] == '':
            print('empty gold: ', i)
    # 计算ROUGE值
    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps=predictions, refs=golds, avg=True, ignore_empty=True)

    # 将数值乘以100
    for key in rouge_score:
        rouge_score[key]['r'] *= 100
        rouge_score[key]['p'] *= 100
        rouge_score[key]['f'] *= 100

    print("ROUGE", 'w/o stopwords' if rmstop else 'w/ stopwords', ': ')
    print(json.dumps(rouge_score, indent=4))


def eval_explain(predict_file):
    print('Start to evaluate the explainability...')
    model, tokenizer = load_model_and_tokenizer(model_type='seq', model_name_or_path=casual_seq_model_path, cache_dir=None)
    lines = read_txt(predict_file)

    logits = seq_predict(model, tokenizer, lines, local_rank)

    # 对logits进行softmax操作以获取概率
    probabilities = torch.nn.functional.softmax(logits, dim=1)

    # 获取标签1的概率
    label_1_probabilities = probabilities[:, 1]

    # 对logits进行argmax操作
    predictions = torch.argmax(logits, dim=1)

    # 统计标签1的总数量
    count_label_1 = (predictions == 1).sum().item()

    # 统计标签1概率大于0.95的数量
    count_high_confidence_label_1 = ((predictions == 1) & (label_1_probabilities > 0.95)).sum().item()

    # 计算标签1概率大于0.95的百分比
    percent_high_confidence_label_1 = (count_high_confidence_label_1 / count_label_1) * 100 if count_label_1 > 0 else 0

    print('Total amount of label 1: ', count_label_1)
    # print('Amount of label 1 with confidence > 95%: ', count_high_confidence_label_1)
    print('Percent of label 1 with confidence > 95%: ', percent_high_confidence_label_1)


if __name__ == '__main__':
    casual_seq_model_path = '/home/yifei/data/unicausal/seq-baseline'
    stopwords_path = '/home/yifei/code/My-Code-Reviewer/evaluator/stopwords.txt'
    local_rank = 0


    predict_file = '/home/yifei/code/code_review_automation/code/generate_predictions/result/predictions_10.txt'
    gold_file = '/home/yifei/code/code_review_automation/code/generate_predictions/golds.txt'

    eval_bleu(predict_file, gold_file, rmstop=False)
    eval_bleu(predict_file, gold_file, rmstop=True)

    eval_rouge(predict_file, gold_file, rmstop=False)
    eval_rouge(predict_file, gold_file, rmstop=True)

    # eval_explain(predict_file)
