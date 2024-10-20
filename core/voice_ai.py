import whisper

class VoiceAI:
    def __init__(self, model: str) -> None:
        self.model = whisper.load_model(model)

    def transcribe(self, audio_data) -> str:
        result = self.model.transcribe(audio_data)
        return result["text"]