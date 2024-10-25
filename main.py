import asyncio
import os
import shutil
import time
import wave
import torch
import numpy as np
import sounddevice as sd
import pygame

# Import the AI modules
from core.config_reader import ConfigReader
from core.logger import Logger, Severity
from core.text_ai import TextAI
from core.tts_ai import TTSAI
from core.voice_ai import VoiceAI

# Audio stream settings
RATE = 48000  # 48 kHz sampling rate (better audio fidelity)
CHANNELS = 1  # Mono audio
CHUNK_SIZE = 2048  # Number of audio samples per frame

# Initialize sound player
pygame.mixer.init()

def play_sound(filename: str):
    """Load and play the sound."""
    sound = pygame.mixer.Sound(filename)
    sound.play()  
    pygame.time.wait(int(sound.get_length() * 1000))

def get_torch_device():
    """Check if a GPU is available and return the appropriate device."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def should_respond(transcription: str, responds_to: list[str]):
    if len(responds_to) == 0:
        return True

    for keyword in responds_to:
        if keyword.lower().replace(" ", "") in transcription.lower().replace(" ", ""):
            return True
        
    return False

def generate_clean_string(input: str):
    chars_to_remove = "\\/:*?\"<>|()',[]-{}"
    sanitized = ''.join(c if c not in chars_to_remove else '' for c in input)
    return sanitized

async def process_speech(text_ai: TextAI, tts_ai: TTSAI, voice_ai: VoiceAI, logger: Logger, keep_audio_files: bool, responds_to: list[str], recording_file="recordings/user_recording.wav"):
    """Process the recorded speech using your AI models."""
    logger.info("Transcribing...")
    transcription = voice_ai.transcribe(recording_file)  # Transcribe audio to text
    logger.info(f"Transcription: {transcription}")

    torch.cuda.empty_cache()

    if transcription == "":
        logger.warning("No message was recorded", Severity.LOW)
        return
    
    command = generate_clean_string(transcription.upper())

    logger.debug(command)
    
    if "WIPE WALTER MEMORY" in command or "TRUNCATE WALTER" in command:
        logger.info(f"Deleted {text_ai.clear_conversation()} messages")
        play_sound("voice_samples/cnvdl.wav")
        return
    
    if not should_respond(transcription, responds_to):
        logger.warning("Keywords for activation not detected. If you think this is a mistake. Please check your 'config.json'", Severity.LOW)
        return
    
    logger.debug(responds_to)

    logger.info("Generating response...")
    response = await text_ai.get_ollama_message(transcription, "user")  # Get response asynchronously
    logger.info(f"AI Response: {response}")

    torch.cuda.empty_cache()

    logger.info("Generating TTS audio...")
    out_file = tts_ai.generate_audio(response)  # Generate speech from response
    logger.info(f"Audio saved as: {out_file}")

    torch.cuda.empty_cache()

    logger.info("Playing response...")
    try:
        # Add sleep to save the file
        time.sleep(0.3)
        play_sound(out_file)  # Play the response audio
        logger.info("Played sound")
    except Exception as e:
        logger.error(f"Something went wrong: {str(e)}", Severity.MEDIUM)

    if not keep_audio_files:
        logger.info("Deleting recordings")
        shutil.rmtree("recordings", True)

def save_audio_to_file(audio_data, logger: Logger, file_path="recordings/user_recording.wav"):
    """Save the audio buffer to a WAV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  

    audio_int16 = normalize_audio(audio_data)
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes for int16 format
        wf.setframerate(RATE)
        wf.writeframes(audio_int16.tobytes())
    logger.info(f"Audio saved as {file_path}")

def normalize_audio(audio_data):
    return (audio_data * 32768).astype(np.int16)  # Convert float32 back to int16

def is_speech(audio_data, energy_threshold_factor=1.5, noise_floor=200):
    """Determine if a chunk contains speech based on energy threshold.
    
    This version calculates a dynamic energy threshold to better ignore noise.
    
    Args:
        audio_data (numpy array): Audio chunk to analyze.
        energy_threshold_factor (float): Factor to multiply the noise floor by to determine the threshold.
        noise_floor (float): A baseline level of noise to consider.
    
    Returns:
        bool: True if speech is detected, False otherwise.
    """
    energy = np.abs(audio_data).mean()  # Calculate average energy level
    
    # Dynamic threshold based on noise floor
    dynamic_threshold = max(noise_floor, energy_threshold_factor * noise_floor)
    
    return energy > dynamic_threshold  # Return True if energy exceeds the dynamic threshold

def get_user_microphone() -> int:
    devices = sd.query_devices()
    # Find the first device with input channels (microphone)
    mics = []
    for device in devices:
        if device['max_input_channels'] > 0:  # Has input channels, likely a microphone
            mics.append({"name": device["name"], "index": device["index"]})
    
    if len(mics) == 0:
        raise RuntimeError("No microphone found")
    
    if len(mics) == 1:
        return mics[0]["index"]
    
    for i, mic in enumerate(mics):
        print(f"{i}: {mic["name"]}")
    
    mic_index = int(input("Please type your microphone number: "))

    return mics[mic_index]["index"]

    

async def main():
    # Get configuration
    logger = Logger("main")
    logger.debug("Debug mode is enabled")

    logger.info("Getting configuration")
    config_reader = ConfigReader("config.json")
    cfg = config_reader.read_config()

    logger.debug(cfg)

    mic_index = None

    if not cfg.always_use_default_mic:
        logger.debug(f"cfg.always_use_default_mic: {cfg.always_use_default_mic}")
        mic_index: int = get_user_microphone()

    device = get_torch_device()
    # Initialize AI modules
    logger.info(f"Using {device.upper()}")

    logger.info("Loading TTS AI...")
    tts_ai = TTSAI(device, voice_file="voice_samples/walter.mp3")

    logger.info("Loading Voice AI...")
    voice_ai = VoiceAI(cfg.whisper_model)

    logger.info("Loading Text AI...")
    text_ai = TextAI(cfg)

    logger.info("Listening for speech...")
    audio_buffer = []
    recording = False

    silence_frames = 0  # Counter for silent frames

    try:
        with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype='int16', blocksize=CHUNK_SIZE, device=mic_index) as stream:
            while True:
                chunk, _ = stream.read(CHUNK_SIZE)

                if recording:
                    # Add the chunk to the audio buffer
                    audio_buffer.append(chunk)

                if is_speech(chunk):
                    silence_frames = 0  # Reset silence counter on speech
                    if not recording:
                        logger.info("Speech detected. Recording...")
                        recording = True
                        # Add the first chunk to the audio buffer
                        audio_buffer.append(chunk)
                
                elif recording:
                    # Increment silence frames since no speech is detected
                    silence_frames += 1

                    if silence_frames >= int(RATE * cfg.grace_period_in_seconds / CHUNK_SIZE):
                        # Grace period passed, stop recording
                        logger.info("Grace period over. Stopping recording...")
                        recording = False
                        silence_frames = 0

                        # Process the recorded audio
                        audio_data = np.concatenate(audio_buffer).astype(np.float32) / 32768.0
                        audio_buffer = []  # Clear buffer

                        recording_file = "recordings/user_recording.wav"

                        save_audio_to_file(audio_data, logger, recording_file)
                        await process_speech(text_ai, tts_ai, voice_ai, logger, cfg.keep_audio_files, cfg.responds_to, recording_file)

    except KeyboardInterrupt:
        logger.info("Stopping...")

if __name__ == "__main__":
    asyncio.run(main())
