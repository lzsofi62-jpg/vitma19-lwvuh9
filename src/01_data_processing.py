import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path("./data")
OUTPUT_CSV = Path("./output/dataset.csv")

def load_all_annotations():
    rows = []

    for json_file in DATA_DIR.glob("*.json"):
        with open(json_file, "r", encoding="utf8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to parse {json_file.name}: {e}")
                continue

        # Ha dict, tegyük listává
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            print(f"WARNING: Unexpected JSON format in {json_file.name}")
            continue

        for item in data:
            try:
                annotations = item.get("annotations", [])
                if not annotations:
                    print(f"WARNING: No annotations in {json_file.name}, item id {item.get('id')}")
                    continue

                result_list = annotations[0].get("result", [])
                if not result_list:
                    print(f"WARNING: Empty result in {json_file.name}, item id {item.get('id')}")
                    continue

                value = result_list[0].get("value", {})

                # rating kiválasztása
                if "choices" in value:
                    rating = value["choices"][0]
                elif "rating" in value:
                    rating = value["rating"]
                else:
                    print(f"WARNING: No 'choices' or 'rating' in item id {item.get('id')}")
                    continue

                # szöveg a JSON-ból
                text_content = item.get("data", {}).get("text", "").strip()
                if not text_content:
                    print(f"WARNING: Empty text in {json_file.name}, item id {item.get('id')}")
                    continue

                rows.append({
                    "text": text_content,
                    "label": rating,
                    "source": json_file.name
                })

            except Exception as e:
                print(f"WARNING: Skipping item in {json_file.name} due to error: {e}")
                continue

    return pd.DataFrame(rows)


def main():
    print("Loading all annotation files...")

    df = load_all_annotations()
    print(f"Loaded {len(df)} samples from {DATA_DIR}")

    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved dataset to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()