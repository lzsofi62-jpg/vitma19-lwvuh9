import pandas as pd
from sklearn.metrics import classification_report
import joblib
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch


DATASET_PATH = "./output/dataset.csv"
BASELINE_PATH = "./output/baseline_model.joblib"
BERT_PATH = "./output/bert_model"    


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    df["label"] = df["label"].astype(str).str.extract(r"(\d)").astype(int) - 1

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    print("\n=== Baseline evaluation ===")
    base = joblib.load(BASELINE_PATH)
    vec = base["vectorizer"]
    clf = base["model"]

    baseline_preds = clf.predict(vec.transform(texts))
    print(classification_report(labels, baseline_preds))

    print("\n=== BERT evaluation ===")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(BERT_PATH)
    model.eval()

    bert_preds = []
    device = torch.device("cpu")
    model.to(device)

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(device)

            outputs = model(**enc)
            pred = outputs.logits.argmax(dim=1).item()
            bert_preds.append(pred)

    print(classification_report(labels, bert_preds))


if __name__ == "__main__":
    main()
