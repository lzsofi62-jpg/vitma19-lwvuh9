import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DistilBertTokenizerFast,
    DistilBertModel,
    AdamW
)

import numpy as np
import re


MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_LEN = 192               
BATCH_SIZE = 4               
EPOCHS = 5                  
LR = 2e-5


def extract_features(text: str):
    tokens = text.split()

    length = len(tokens)
    avg_token_len = sum(len(t) for t in tokens) / max(1, length)

    num_numbers = sum(t.isdigit() for t in tokens)
    num_links = len(re.findall(r"http|www", text.lower()))
    num_par_refs = text.count("§") + text.lower().count("bek")
    num_upper = sum(1 for t in tokens if t.isupper())

    return np.array([
        length,
        avg_token_len,
        num_numbers,
        num_links,
        num_par_refs
    ], dtype=np.float32)

class ASZFDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # BERT tokenizálás
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}

        item["extra_features"] = torch.tensor(
            extract_features(text),
            dtype=torch.float
        )

        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class BertWithFeatures(nn.Module):
    def __init__(self, num_labels=5, feature_dim=5):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)

        bert_hidden = 768

        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden + feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, extra_features, labels=None):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = bert_out.last_hidden_state[:, 0]  # [CLS]

        x = torch.cat([cls, extra_features], dim=1)
        logits = self.classifier(x)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits

        return logits


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        features = batch["extra_features"].to(device)
        labels = batch["labels"].to(device)

        loss, logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            extra_features=features,
            labels=labels
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    print("Loading data...")
    df = pd.read_csv("output/dataset.csv")

    # label 1–5 → 0–4 formátum
    df["label"] = df["label"].astype(str).str.extract(r"(\d)").astype(int) - 1

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    dataset = ASZFDataset(texts, labels, tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cpu")
    print("Using CPU.")

    model = BertWithFeatures(num_labels=5).to(device)

    optimizer = AdamW(model.parameters(), lr=LR)

    print("Training...")
    for e in range(EPOCHS):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"Epoch {e+1}/{EPOCHS} - Loss: {loss:.4f}")

    print("Saving model...")
    save_path = "./output/bert_with_features/"
    tokenizer.save_pretrained(save_path)
    torch.save(model.state_dict(), save_path + "model.pt")

    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
