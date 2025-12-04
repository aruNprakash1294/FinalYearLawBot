import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_all_cases():
    dfs = []

    try:
        dfs.append(pd.read_csv("cases.csv"))
        print("Loaded cases.csv")
    except FileNotFoundError:
        print("cases.csv not found, skipping.")

    try:
        dfs.append(pd.read_csv("hf_cases.csv"))
        print("Loaded hf_cases.csv")
    except FileNotFoundError:
        print("hf_cases.csv not found, skipping.")

    if not dfs:
        raise RuntimeError("No case CSV files found!")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total cases loaded: {len(df)}")
    return df

def build_index():
    df = load_all_cases()

    model_name = "law-ai/InLegalBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    texts = []
    metadata = []

    for _, row in df.iterrows():
        text = f"Facts: {row['facts']} Decision: {row['decision_summary']}"
        texts.append(text)
        metadata.append({
            "id": int(row["id"]),
            "title": row["title"],
            "category": row["category"],
            "facts": row["facts"],
            "decision_summary": row["decision_summary"],
            "court": row.get("court", ""),
            "year": row.get("year", ""),
            "provisions": row.get("provisions", "")
        })

    all_embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_emb.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)

    index = {
        "embeddings": embeddings,
        "metadata": metadata,
    }
    torch.save(index, "case_index.pth")
    print("Saved index to case_index.pth")
    print("Embeddings shape:", embeddings.shape)

if __name__ == "__main__":
    build_index()
