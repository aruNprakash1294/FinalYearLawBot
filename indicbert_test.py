from transformers import AlbertTokenizer, AutoModel
import torch

# Load IndicBERT for Tamil
model_name = "ai4bharat/indic-bert"
tokenizer = AlbertTokenizer.from_pretrained(model_name)  # Use slow tokenizer explicitly
model = AutoModel.from_pretrained(model_name)

# Sample Tamil query
text = "உங்கள் சட்ட உதவி தேவை"

# Tokenize input
inputs = tokenizer(text, return_tensors="pt")

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Use [CLS] token embedding for classification
cls_embedding = outputs.last_hidden_state[:, 0, :]
print("CLS Embedding shape:", cls_embedding.shape)
