import os
from TTS.api import TTS


class TTSAI:
    def __init__(self, torch_device: str, lang = "en", voice_file = "voice_samples/walter.wav", tts_model = "tts_models/multilingual/multi-dataset/xtts_v2") -> None:
        self.lang = lang
        self.voice_file = voice_file
        self.tts = TTS(tts_model).to(torch_device)

    def __sanitize_filename(self, filename: str, max_length: int = 150) -> str:
        # Replace invalid characters with hyphens and truncate to max_length
        invalid_chars = '\\/:*?"<>|()[]{} '
        sanitized = ''.join(c if c not in invalid_chars else '-' for c in filename)
        return sanitized[:max_length]

    def generate_audio(self, text: str):
        try:
            directory = "recordings"

            filename = self.__sanitize_filename(text[:150], 50)
            output_file = os.path.join(directory, f"{filename}.wav")

            if os.path.exists(output_file):
                os.remove(output_file)

            if not os.path.exists(directory):
                os.mkdir(directory)

            self.tts.tts_to_file(text=text, speaker_wav=self.voice_file, language=self.lang, file_path=output_file)

            return output_file
        except Exception as e:
            print("Could not generate: " + str(e))
            return ""