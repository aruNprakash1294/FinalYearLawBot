import whisper
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# ---------------------------------
# 0. Device
# ---------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------
# 1. Load models once
# ---------------------------------
print("Loading Whisper model...")
whisper_model = whisper.load_model("small").to(device)

print("Loading Indic sentence-BERT model...")
indic_model = SentenceTransformer("l3cube-pune/indic-sentence-bert-nli")
indic_model.to(device)

print("Loading InLegalBERT model...")
legal_model_name = "law-ai/InLegalBERT"
legal_tokenizer = AutoTokenizer.from_pretrained(legal_model_name)
legal_model = AutoModel.from_pretrained(legal_model_name).to(device)


# ---------------------------------
# 2. Helper functions
# ---------------------------------
def transcribe_tamil_audio(audio_path: str) -> str:
    """
    Use Whisper to transcribe Tamil audio to text.
    """
    print(f"\n[Whisper] Transcribing audio: {audio_path}")
    result = whisper_model.transcribe(audio_path, language="ta")
    text = result.get("text", "").strip()
    print(f"[Whisper] Transcription: {text}")
    return text


def get_tamil_embedding(text: str) -> torch.Tensor:
    """
    Use Indic sentence-BERT to get an embedding for Tamil text.
    """
    print(f"\n[IndicBERT] Encoding Tamil text:\n  {text}")
    emb = indic_model.encode([text], convert_to_tensor=True, device=device)
    print(f"[IndicBERT] Embedding shape: {emb.shape}")
    return emb


def get_legal_embedding(text: str) -> torch.Tensor:
    """
    Use InLegalBERT to get an embedding for English legal text.
    """
    print(f"\n[InLegalBERT] Encoding legal English text:\n  {text}")
    inputs = legal_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = legal_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    print(f"[InLegalBERT] CLS embedding shape: {cls_embedding.shape}")
    return cls_embedding


def load_case_index(path: str = "case_index.pth"):
    """
    Load precomputed case embeddings and metadata.
    """
    data = torch.load(path)
    embeddings = data["embeddings"]  # [N, d]
    metadata = data["metadata"]      # list of dicts
    return embeddings, metadata


def find_similar_cases(query_text: str, top_k: int = 3):
    """
    Given an English legal query text, embed it with InLegalBERT
    and find the most similar cases.
    """
    print(f"\n[Similar Cases] Finding similar cases for query:\n  {query_text}")

    # 1. Query embedding
    query_emb = get_legal_embedding(query_text).cpu()   # [1, d]

    # 2. Case index
    case_embs, meta = load_case_index()                 # [N, d], list[dict]

    # 3. Cosine similarity
    sims = F.cosine_similarity(query_emb, case_embs)    # [N]
    topk = torch.topk(sims, k=min(top_k, len(meta)))
    indices = topk.indices.tolist()
    scores = topk.values.tolist()

    print(f"\nTop {len(indices)} similar cases:\n")
    results = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        case = meta[idx]
        print(f"{rank}. {case['title']} (similarity: {score:.3f})")
        print(f"   Category: {case['category']}")
        print(f"   Court: {case.get('court', 'N/A')}  Year: {case.get('year', 'N/A')}")
        print(f"   Facts: {case['facts']}")
        print(f"   Decision: {case['decision_summary']}\n")
        results.append({"case": case, "score": score})

    return results


# ---------------------------------
# 3. Main pipeline
# ---------------------------------
def main():
    # Step 1: Transcribe Tamil audio
    audio_path = "sample_audio.m4a"  # change if needed
    tamil_text = transcribe_tamil_audio(audio_path)

    if not tamil_text:
        print("\n[ERROR] No text transcribed from audio. Check your audio file.")
        return

    # Step 2: Get Tamil embedding (for future extensions)
    tamil_emb = get_tamil_embedding(tamil_text)

    # Step 3: English version of the query
    # For now we manually write one matching your intimidation scenario.
    english_query = (
        "What remedies are available if the police threaten me and say "
        "they will file a false case against me?"
    )

    # Step 4: Find similar cases
    similar_cases = find_similar_cases(english_query, top_k=3)

    print("\n✅ Pipeline run complete.")
    print("   - Whisper transcription ✔")
    print("   - Indic sentence-BERT embedding ✔")
    print("   - InLegalBERT case similarity search ✔")


if __name__ == "__main__":
    main()
