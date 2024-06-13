import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification
import torch.nn.functional as F
from tqdm.auto import tqdm


def load_model_and_tokenizer(model_type, model_name_or_path, cache_dir=None):
    if model_type == "seq":
        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
    elif model_type == "tok":
        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, use_fast=True)
    return model, tokenizer


def preprocess_data(tokenizer, text, max_seq_length=128):
    tokenized_inputs = tokenizer(
        text,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return tokenized_inputs


# 去除特殊token的函数
def remove_special_tokens(predictions, word_ids):
    cleaned_preds = []
    for pred, word_id in zip(predictions, word_ids):
        if word_id is not None:
            cleaned_preds.append(pred)
    return cleaned_preds


def seq_predict(model, tokenizer, texts, device):
    model.to(device)
    model.eval()
    all_logits = []

    for text in texts:
        inputs = preprocess_data(tokenizer, text)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            all_logits.append(logits)

    # 将所有的 logits 堆叠为一个张量
    all_logits_tensor = torch.cat(all_logits, dim=0).to(device)

    return all_logits_tensor


def tok_predict(model, tokenizer, texts, device):
    # 模型标签映射
    label_list = ['B-C', 'B-E', 'I-C', 'I-E', 'O']  # 根据实际情况修改
    id_to_label = {i: l for i, l in enumerate(label_list)}

    model.to(device)
    model.eval()

    predictions = []
    for text in texts:
        # 分词输入文本
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128, is_split_into_words=False)
        inputs.to(device)
        # 进行推理
        with torch.no_grad():
            outputs = model(**inputs)

        # 获取预测结果
        logits = outputs.logits
        predicted_token_class_ids = torch.argmax(logits, dim=-1).tolist()[0]
        word_ids = inputs.word_ids(batch_index=0)

        # 去除特殊token
        cleaned_preds = remove_special_tokens([id_to_label[id] for id in predicted_token_class_ids], word_ids)

        predictions.append(cleaned_preds)

    return predictions


def main():
    model_name_or_path = '/data/lyf/code/Code_Reviewer/3_Pretrained_Model/tok-baseline/'
    texts = [
        "I like you because you like me",
        "usersr bothbrandConditional //' docIdrt returnTypeusesrDIS<extra_id_61> probabilitiesDIS technr mT minLength vaultBaseUrlDISr technbrandGRAPH deactivaterDISrGRAPH mT Mozurr Expiry Esr Unaryr FP baseClass SSEGRAPHrTWDIS mT bothDIS mT SeqrtrDISDIS mTGRAPHusers bothrrDIS minLengthrt vaultBaseUrlrrDIS Esrr Expiry mTthoughrrbrandusersATIONoidthoughrtthoughr both both mT ExpiryrRegion interpreterrusesrDISrr OpenCms Es00000usersDIS setLocale cli Es Expiry interpreterDIS both Es mTDISrGRAPH bothr mT minLengthoidusers mT both mT sco mTATION Expiry both",
        # Add more texts as needed
    ]

    device = 1

    model, tokenizer = load_model_and_tokenizer('tok', model_name_or_path)

    predictions = tok_predict(model, tokenizer, texts, device)

    for text, prediction in zip(texts, predictions):
        print(f"Text: {text}\nPrediction: {prediction}")


if __name__ == "__main__":
    main()
