"""Whisper listening.

Before a track goes on air, a slice from the middle of it is transcribed with
whisper.cpp.  That gives the DJ something real to react to — the actual words
and the language actually being sung — instead of only the file's tags.

Results are cached in the library so a track is only ever listened to once.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from . import config
from .logging_util import get_logger

log = get_logger(__name__)

_LANG_RE = re.compile(r"auto-detected language:\s*([a-z]{2})", re.I)
_NOISE = re.compile(r"\[[^\]]*\]|\([^)]*\)|♪|\*")


def available() -> bool:
    return (
        config.WHISPER_ENABLED
        and Path(config.WHISPER_BIN).is_file()
        and Path(config.WHISPER_MODEL).is_file()
    )


def _extract_slice(path: str, duration: float) -> Path | None:
    """16 kHz mono wav from a musically interesting part of the track."""
    start = max(0.0, duration * 0.32) if duration > 60 else max(0.0, duration * 0.2)
    length = config.WHISPER_SAMPLE_SECONDS
    tmp = Path(tempfile.mkstemp(prefix="gemradio_listen_", suffix=".wav")[1])
    cmd = [
        config.FFMPEG, "-v", "quiet", "-nostdin", "-y",
        "-ss", f"{start:.2f}", "-t", f"{length:.2f}", "-i", path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(tmp),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=90)
    except Exception:
        tmp.unlink(missing_ok=True)
        return None
    if not tmp.is_file() or tmp.stat().st_size < 8000:
        tmp.unlink(missing_ok=True)
        return None
    return tmp


def transcribe(path: str, duration: float) -> tuple[str, str]:
    """Return (text, language_code).  Empty text means instrumental or failed."""
    if not available():
        return "", ""
    wav = _extract_slice(path, duration)
    if wav is None:
        return "", ""
    # `-np` would also hide the auto-detected language line, which is the whole
    # point of listening: stdout stays clean without it.
    cmd = [
        str(config.WHISPER_BIN),
        "-m", str(config.WHISPER_MODEL),
        "-f", str(wav),
        "-l", "auto",
        "-t", str(config.WHISPER_THREADS),
        "-nt",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=300, text=True)
    except Exception:
        log.debug("whisper failed on %s", path, exc_info=True)
        return "", ""
    finally:
        wav.unlink(missing_ok=True)

    lang = ""
    m = _LANG_RE.search(proc.stderr or "")
    if m:
        lang = m.group(1).lower()

    text = _NOISE.sub(" ", proc.stdout or "")
    text = re.sub(r"\s+", " ", text).strip()

    # Whisper hallucinates stock phrases on instrumental music; drop those.
    if len(text) < 12:
        return "", lang
    lowered = text.lower()
    junk = ("thank you for watching", "subtitles by", "amara.org",
            "thanks for watching", "sous-titres", "sottotitoli")
    if any(j in lowered for j in junk):
        return "", lang
    words = text.split()
    if len(words) > 4 and len(set(words)) <= max(2, len(words) // 6):
        return "", lang            # degenerate repetition

    return text[:600], lang
