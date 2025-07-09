from transformers import pipeline
classifier = pipeline("sentiment-analysis")
print(classifier("I love final year projects!"))
