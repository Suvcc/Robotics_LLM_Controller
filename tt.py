import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000  # Whisper expects 16kHz
DURATION = 5  # seconds

print("Recording... speak now")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
write("recorded.wav", SAMPLE_RATE, audio)
print("Done recording")

model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe("recorded.wav")

for segment in segments:
    print(f"Transcribed: {segment.text}")