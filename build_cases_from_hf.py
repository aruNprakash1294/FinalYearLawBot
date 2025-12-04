from datasets import load_dataset
import csv
import re

# Keywords to roughly detect police-related cases
KEYWORDS = [
    "police", "officer", "constable", "sub-inspector", "sub inspector",
    "si ", "psi ", "custody", "custodial", "lock-up", "lockup",
    "detention", "station", "police station"
]

MAX_CASES = 50  # limit so it's light for your project

def is_police_related(text: str) -> bool:
    text_low = text.lower()
    return any(k in text_low for k in KEYWORDS)

def split_summary(summary: str):
    # Split into rough sentences
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())
    if not sentences:
        return summary, summary

    # First 1–2 sentences as "facts", rest as "decision"
    facts = " ".join(sentences[:2])
    decision = " ".join(sentences[2:]) if len(sentences) > 2 else summary
    return facts, decision

def extract_year(text: str):
    match = re.search(r'(19|20)\d{2}', text)
    return match.group(0) if match else ""

def main():
    print("Loading subset of ninadn/indian-legal from Hugging Face...")
    # Only take first 2000 to keep it light
    ds = load_dataset("ninadn/indian-legal", split="train[:2000]")

    rows = []
    case_id = 1

    for ex in ds:
        full_text = ex["Text"]
        summary = ex["Summary"]

        combined = full_text + " " + summary
        if not is_police_related(combined):
            continue

        facts, decision = split_summary(summary)
        year = extract_year(full_text)

        # Simple title: first sentence of summary (trimmed)
        title_sentence = re.split(r'(?<=[.!?])\s+', summary.strip())[0]
        title = title_sentence[:80]

        row = {
            "id": case_id,
            "title": title,
            "category": "police_related",
            "facts": facts,
            "decision_summary": decision,
            "court": "",  # this dataset doesn't give court explicitly
            "year": year,
            "provisions": ""
        }
        rows.append(row)
        case_id += 1

        if case_id > MAX_CASES:
            break

    print(f"Collected {len(rows)} police-related cases.")

    out_file = "hf_cases.csv"
    fieldnames = ["id", "title", "category", "facts",
                  "decision_summary", "court", "year", "provisions"]

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
