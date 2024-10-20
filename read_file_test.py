import whisper

model = whisper.load_model("base")
result = model.transcribe("conversation.wav")
print(result["text"])