"""Speech-to-text input source.

Contract with the rest of the system: `listen()` returns the transcribed user
command as plain text. The CLI feeds that text into the same pipeline as typed
input — nothing downstream knows it came from a microphone.
"""
from pathlib import Path

import sounddevice as sd
from faster_whisper import WhisperModel
from scipy.io.wavfile import write

WAV_PATH = "logs/last_recording.wav"

# Whisper is cached after the first call — reloading it inside STT() would
# cost several seconds on every voice command.
_model: WhisperModel | None = None
_model_size: str | None = None


def _get_model(model_size: str) -> WhisperModel:
    global _model, _model_size
    if _model is None or _model_size != model_size:
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _model_size = model_size
    return _model


def combine_segments(segments) -> str:
    """Join Whisper segments into a single one-line command string."""
    return " ".join(seg.text.strip() for seg in segments).strip()


def recordVoice(file_path=WAV_PATH, SAMPLE_RATE=16000, DURATION=5):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    print("Start speaking: ")
    sd.wait()
    write(file_path, SAMPLE_RATE, audio)
    print("Stopped Recording.")


def STT(file_path=WAV_PATH, model_size="base", language=None) -> str:
    model = _get_model(model_size)
    segments, info = model.transcribe(file_path, language=language)
    return combine_segments(segments)


def listen(cfg=None) -> str:
    """Record one utterance and return it as text. This is what the CLI calls.

    `cfg` is an aliengo.config.SpeechConfig; defaults are used when omitted.
    """
    duration = cfg.duration_s if cfg else 5
    model_size = cfg.model_size if cfg else "base"
    language = cfg.language if cfg else None
    recordVoice(DURATION=duration)
    return STT(model_size=model_size, language=language)
