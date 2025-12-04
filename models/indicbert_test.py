from sentence_transformers import SentenceTransformer
import torch

model_name = "l3cube-pune/indic-sentence-bert-nli"
model = SentenceTransformer(model_name)

# Sample Tamil query
text = "உங்கள் சட்ட உதவி தேவை"

# SentenceTransformer takes a list of sentences
emb = model.encode([text], convert_to_tensor=True)

print("Embedding shape:", emb.shape)  # should be [1, 768]

