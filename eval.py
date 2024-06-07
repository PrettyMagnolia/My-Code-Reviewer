import os

import nltk
from rouge import Rouge


def read_txt_files(pred_path, gold_path, add_space):
    with open(pred_path, 'r', encoding='utf-8') as file:
        pred_lines = file.readlines()

    with open(gold_path, 'r', encoding='utf-8') as file:
        gold_lines = file.readlines()

    pred_nls, golds = pred_lines, gold_lines
    for i in range(len(pred_nls)):
        chars = "(_)`."
        for c in chars:
            pred_nls[i] = pred_nls[i].replace(c, " " + c + " ")
            pred_nls[i] = " ".join(pred_nls[i].split())
            golds[i] = golds[i].replace(c, " " + c + " ")
            golds[i] = " ".join(golds[i].split())

    # 对预测结果和参考答案进行分词处理
    predictions = [" ".join(nltk.wordpunct_tokenize(pred_nls[i])) for i in range(len(pred_nls))]
    golds = [" ".join(nltk.wordpunct_tokenize(g)) for g in golds]

    if add_space:
        for i in range(len(predictions)):
            if predictions[i] == '':
                predictions[i] = ' '
            if golds[i] == '':
                golds[i] = ' '

    return predictions, golds


def remove_stop(predictions, golds):
    # 读取停用词文件中的停用词列表
    stopwords = open(os.path.join("./evaluator", "stopwords.txt")).readlines()
    stopwords = [stopword.strip() for stopword in stopwords]
    # 对参考答案和预测结果中的单词进行停用词移除
    golds = [" ".join([word for word in ref.split() if word not in stopwords]) for ref in golds]
    predictions = [" ".join([word for word in hyp.split() if word not in stopwords]) for hyp in predictions]

    for i in range(len(predictions)):
        if predictions[i] == '':
            predictions[i] = ' '
        if golds[i] == '':
            golds[i] = ' '

    return predictions, golds


def rouge_l(pred, gold):
    """
    计算ROUGE-L值。

    参数:
    pred (list): 预测摘要列表
    gold (list): 真实摘要列表

    返回:
    dict: 包含ROUGE-L值的字典
    """
    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps=pred, refs=gold, avg=True)
    return rouge_score


def perfect_prediction(pred, gold):
    res = 0
    for i in range(len(pred)):
        if pred[i] == gold[i]:
            print(pred[i], gold[i])
            res += 1
    return res / len(pred)

if __name__ == '__main__':
    check = 'origin_topk10'
    pred_path = '/data/lyf/code/Code_Reviewer/0_Result/preds/preds_{}.txt'.format(check)
    gold_path = '/data/lyf/code/Code_Reviewer/0_Result/golds.txt'
    pred, gold = read_txt_files(pred_path, gold_path, True)
    pred_, gold_ = remove_stop(pred, gold)
    print(perfect_prediction(pred, gold))
    print(perfect_prediction(pred_, gold_))
    # # 有停用词
    # rouge_score = rouge_l(pred, gold)
    # print("有停用词", rouge_score)
    # for i in range(len(pred_)):
    #     if len(pred_[i]) <= 1:
    #         pred_[i] += ' '
    # # 无停用词
    # rouge_score_ = rouge_l(pred_, gold_)
    # print("无停用词", rouge_score_)
