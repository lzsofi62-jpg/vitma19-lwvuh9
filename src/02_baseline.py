import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

DATASET_PATH = "./output/dataset.csv"
MODEL_OUT = "./output/baseline_model.joblib"

def main():
    df = pd.read_csv(DATASET_PATH)

    # label konvertálás 1–5 → 0–4
    df["label"] = df["label"].astype(str).str[0].astype(int) - 1

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    print("TF-IDF vectorizer fitting...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training baseline Logistic Regression...")
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    print("\n=== Baseline evaluation ===")
    print(classification_report(y_test, preds))

    Path("./output").mkdir(exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": clf}, MODEL_OUT)
    print(f"Baseline model saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
