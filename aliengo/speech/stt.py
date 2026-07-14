"""Speech-to-text input source.

Contract with the rest of the system: `listen()` returns the transcribed user
command as plain text. The CLI feeds that text into the same pipeline as typed
input — nothing downstream knows it came from a microphone.

Add your imports (sounddevice, scipy, faster_whisper, ...) at the top and your
logic inside recordVoice/STT below. Tip: if you load a WhisperModel, load it
once at module level (or cache it), not inside STT() — reloading it per call
costs several seconds on every command.
"""
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from faster_whisper import WhisperModel


WAV_PATH = "logs/last_recording.wav"


def recordVoice(file_path=WAV_PATH,SAMPLE_RATE =16000, DURATION=5):
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    print("Start speaking: ")
    sd.wait()
    write(file_path, SAMPLE_RATE, audio)
    print("Stopped Recording.")


def STT(file_path=WAV_PATH) -> str:
    
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(file_path)
    combinedTxts= ""
    for segment in segments:
        combinedTxts+=segment.text + "\n"

    return combinedTxts


def listen() -> str:
    """Record one utterance and return it as text. This is what the CLI calls."""
    recordVoice()
    return STT().strip()
