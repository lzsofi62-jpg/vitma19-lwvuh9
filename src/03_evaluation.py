import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from pathlib import Path

MODEL_PATH = "/app/output/model"
DATASET_PATH = "/app/output/dataset.csv"
REPORT_PATH = "/app/output/evaluation.txt"

class TextDataset:
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        return enc, row["label"] - 1  # 0–4

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(DATASET_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

    dataset = TextDataset(df, tokenizer)
    loader = DataLoader(dataset, batch_size=8)

    true_labels = []
    pred_labels = []

    model.eval()
    for inputs, labels in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs).logits
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        pred_labels.extend(preds)
        true_labels.extend(labels)

    report = classification_report(true_labels, pred_labels)
    Path("/app/output").mkdir(exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print("Evaluation complete. Report saved.")

if __name__ == "__main__":
    main()
