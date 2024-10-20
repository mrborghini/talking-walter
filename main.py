import asyncio
import shutil
import torch
import numpy as np
import pyaudio
import webrtcvad
from playsound import playsound

# Import your AI modules
from core.config_reader import ConfigReader
from core.logger import Logger, Severity
from core.text_ai import TextAI
from core.tts_ai import TTSAI
from core.voice_ai import VoiceAI

# Audio stream settings
RATE = 16000  # 16 kHz sampling rate (required by WebRTC VAD)
CHANNELS = 1  # Mono audio
CHUNK_SIZE = 640 # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # 16-bit PCM format

# Initialize WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(3)  # 0-3 (0: most aggressive, 3: least)

# Initialize PyAudio
p = pyaudio.PyAudio()

def get_device():
    """Check if a GPU is available and return the appropriate device."""
    return "cuda" if torch.cuda.is_available() else "cpu"

async def process_speech(audio_data, text_ai: TextAI, tts_ai: TTSAI, voice_ai: VoiceAI, logger: Logger, keep_audio_files: bool):
    """Process the recorded speech using your AI models."""
    logger.info("Transcribing...")
    transcription = voice_ai.transcribe(audio_data)  # Transcribe audio to text
    logger.info(f"Transcription: {transcription}")

    logger.info("Generating response...")
    response = await text_ai.get_ollama_message(transcription, "User")  # Get response asynchronously
    logger.info(f"AI Response: {response}")

    logger.info("Generating TTS audio...")
    out_file = tts_ai.generate_audio(response)  # Generate speech from response
    logger.info(f"Audio saved as: {out_file}")

    logger.info("Playing response...")
    try:
        playsound(out_file)  # Play the response audio
        logger.info("Played sound")
    except Exception as e:
        logger.error(f"Something went wrong: {str(e)}", Severity.MEDIUM)

    if not keep_audio_files:
        shutil.rmtree("recordings", True)

def is_speech(chunk, energy_threshold=300):
    audio_data = np.frombuffer(chunk, dtype=np.int16)
    energy = np.abs(audio_data).mean()  # Calculate average energy level

    return energy > energy_threshold  # Return True only if energy exceeds threshold


async def main():
    # Get configuration
    logger = Logger("main")

    logger.info("Getting configuration")

    config_reader = ConfigReader("config.json")
    cfg = config_reader.read_config()
    # Initialize AI modules
    device = get_device()
    logger.info(f"Using {device.upper()}")

    logger.info("Loading TTS AI...")
    tts_ai = TTSAI(device)

    logger.info("Loading Voice AI...")
    voice_ai = VoiceAI(cfg.whisper_model)

    logger.info("Loading Text AI...")
    text_ai = TextAI(cfg)

    # Open audio stream
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    logger.info("Listening for speech...")
    audio_buffer = []
    recording = False

    silence_frames = 0  # Counter for silent frames
    try:
        while True:
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            if is_speech(chunk):
                silence_frames = 0  # Reset silence counter on speech
                if not recording:
                    logger.info("Speech detected. Recording...")
                    recording = True

                # Add chunk to the audio buffer
                audio_buffer.append(np.frombuffer(chunk, dtype=np.int16))

            elif recording:
                # Increment silence frames since no speech is detected
                silence_frames += 1

                if silence_frames >= cfg.grace_period_frames: # Number of silent frames to wait before stopping (e.g., 0.5s if CHUNK_SIZE=320 at 16kHz)
                    # Grace period passed, stop recording
                    logger.info("Grace period over. Stopping recording...")
                    recording = False
                    silence_frames = 0

                    # Process the recorded audio
                    audio_data = np.concatenate(audio_buffer).astype(np.float32) / 32768.0
                    audio_buffer = []  # Clear buffer

                    await process_speech(audio_data, text_ai, tts_ai, voice_ai, logger, cfg.keep_audio_files)

    except KeyboardInterrupt:
        logger.info("\nStopping...")
    finally:
        # Close the stream gracefully
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    asyncio.run(main())
