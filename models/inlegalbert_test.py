from transformers import AutoTokenizer, AutoModel
import torch

model_name = "law-ai/InLegalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "This agreement is governed by the Indian Contract Act, 1872."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
cls_embedding = outputs.last_hidden_state[:, 0, :]

print("CLS Embedding shape:", cls_embedding.shape)
