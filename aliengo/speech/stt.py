"""Speech-to-text input source.

Contract with the rest of the system: `listen()` returns the transcribed user
command as plain text. The CLI feeds that text into the same pipeline as typed
input — nothing downstream knows it came from a microphone.
"""
from pathlib import Path
from tempfile import NamedTemporaryFile
import threading

import sounddevice as sd
from faster_whisper import WhisperModel
from scipy.io.wavfile import write

WAV_PATH = "logs/last_recording.wav"

# Whisper is cached after the first call — reloading it inside STT() would
# cost several seconds on every voice command.
_model: WhisperModel | None = None
_model_size: str | None = None
_model_lock = threading.RLock()

UPLOAD_SUFFIXES = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
}


def _get_model(model_size: str) -> WhisperModel:
    global _model, _model_size
    with _model_lock:
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
    with _model_lock:
        segments, info = model.transcribe(file_path, language=language)
        return combine_segments(segments)


def _duration_seconds(file_path: str) -> float:
    """Read duration with PyAV, including recorder blobs lacking metadata."""
    import av

    with av.open(file_path) as container:
        if container.duration is not None:
            return float(container.duration / av.time_base)
        audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio_stream is None:
            raise ValueError("Uploaded file has no audio stream.")
        if audio_stream.duration is not None and audio_stream.time_base is not None:
            return float(audio_stream.duration * audio_stream.time_base)
        duration = 0.0
        for frame in container.decode(audio=0):
            if frame.pts is not None and frame.time_base is not None:
                duration = max(
                    duration,
                    float(frame.pts * frame.time_base)
                    + (frame.samples / frame.sample_rate),
                )
        return duration


def transcribe_upload(data: bytes, content_type: str, config) -> dict:
    """Transcribe a short browser recording and always remove the temp file."""
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    suffix = UPLOAD_SUFFIXES.get(media_type)
    if not suffix:
        raise ValueError(f"Unsupported audio type: {media_type or 'unknown'}.")

    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(prefix="aliengo-upload-", suffix=suffix, delete=False) as f:
            f.write(data)
            temp_path = Path(f.name)
        duration = _duration_seconds(str(temp_path))
        if duration <= 0:
            raise ValueError("Uploaded audio is empty.")
        if duration > config.server.max_audio_duration_s:
            raise ValueError(
                f"Audio is {duration:.1f}s; maximum is "
                f"{config.server.max_audio_duration_s:.1f}s."
            )
        text = STT(
            str(temp_path),
            model_size=config.speech.model_size,
            language=config.speech.language,
        )
        return {"text": text, "duration_s": round(duration, 2)}
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


def listen(cfg=None) -> str:
    """Record one utterance and return it as text. This is what the CLI calls.

    `cfg` is an aliengo.config.SpeechConfig; defaults are used when omitted.
    """
    duration = cfg.duration_s if cfg else 5
    model_size = cfg.model_size if cfg else "base"
    language = cfg.language if cfg else None
    recordVoice(DURATION=duration)
    return STT(model_size=model_size, language=language)
