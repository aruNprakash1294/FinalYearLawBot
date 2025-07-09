import whisper

# Load the medium Whisper model
model = whisper.load_model("medium")

# Transcribe the Tamil audio file
result = model.transcribe("sample_tamil.mp3", language="ta")

# Print the transcription
print("Transcription:")
print(result["text"])
