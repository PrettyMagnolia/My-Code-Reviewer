import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
from tqdm.auto import tqdm


def load_model_and_tokenizer(model_name_or_path, cache_dir=None):
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
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


def predict(model, tokenizer, texts, device):
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


def main():
    model_name_or_path = '/home/liuyifei@corp.sse.tongji.edu.cn/code_reviewer/seq-baseline'
    texts = [
        "I like you because you like me",
        "usersr bothbrandConditional //' docIdrt returnTypeusesrDIS<extra_id_61> probabilitiesDIS technr mT minLength vaultBaseUrlDISr technbrandGRAPH deactivaterDISrGRAPH mT Mozurr Expiry Esr Unaryr FP baseClass SSEGRAPHrTWDIS mT bothDIS mT SeqrtrDISDIS mTGRAPHusers bothrrDIS minLengthrt vaultBaseUrlrrDIS Esrr Expiry mTthoughrrbrandusersATIONoidthoughrtthoughr both both mT ExpiryrRegion interpreterrusesrDISrr OpenCms Es00000usersDIS setLocale cli Es Expiry interpreterDIS both Es mTDISrGRAPH bothr mT minLengthoidusers mT both mT sco mTATION Expiry both",
        # Add more texts as needed
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, tokenizer = load_model_and_tokenizer(model_name_or_path)

    predictions = predict(model, tokenizer, texts, device)

    for text, prediction in zip(texts, predictions):
        print(f"Text: {text}\nPrediction: {prediction}")


if __name__ == "__main__":
    main()
