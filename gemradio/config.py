"""Central configuration for GemRadio.

Everything is auto-detected from the machine, with environment overrides so the
station can be re-pointed without touching code. No network access anywhere.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path.home()


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(os.path.expanduser(raw)) if raw else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
STATE_DIR = _env_path("GEMRADIO_STATE_DIR", HOME / ".local" / "share" / "gemradio")
CACHE_DIR = _env_path("GEMRADIO_CACHE_DIR", HOME / ".cache" / "gemradio")
DB_PATH = STATE_DIR / "library.db"
COVER_DIR = CACHE_DIR / "covers"
TTS_DIR = CACHE_DIR / "tts"
LOG_PATH = STATE_DIR / "gemradio.log"


def _detect_music_root() -> Path:
    candidates = [
        HOME / "Music" / "מוזיקה",
        HOME / "Music",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return HOME / "Music"


MUSIC_ROOT = _env_path("GEMRADIO_MUSIC_DIR", _detect_music_root())

AUDIO_EXTENSIONS = {
    ".mp3", ".flac", ".m4a", ".mp4", ".aac", ".ogg", ".oga", ".opus",
    ".wav", ".wma", ".aiff", ".aif", ".alac", ".ape", ".mpc",
}

IMAGE_NAME_HINTS = (
    "cover", "folder", "front", "albumart", "album", "artwork", "case",
)

# --------------------------------------------------------------------------
# External binaries
# --------------------------------------------------------------------------

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

PIPER_SITE = _env_path("GEMRADIO_PIPER_SITE", HOME / "piper-gpu" / "site")
PIPER_VOICES_DIR = _env_path(
    "GEMRADIO_PIPER_VOICES", HOME / "Voice_typing" / "piper_voices"
)
PIPER_USE_CUDA = _env_flag("GEMRADIO_PIPER_CUDA", False)

WHISPER_BIN = _env_path(
    "GEMRADIO_WHISPER_BIN", HOME / "whisper.cpp" / "build" / "bin" / "whisper-cli"
)
WHISPER_MODEL = _env_path(
    "GEMRADIO_WHISPER_MODEL", HOME / "whisper.cpp" / "models" / "ggml-small.bin"
)
WHISPER_ENABLED = _env_flag("GEMRADIO_WHISPER", True)
WHISPER_THREADS = int(_env_float("GEMRADIO_WHISPER_THREADS", min(8, os.cpu_count() or 4)))
WHISPER_SAMPLE_SECONDS = _env_float("GEMRADIO_WHISPER_SECONDS", 20.0)

OLLAMA_HOST = os.environ.get("GEMRADIO_OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("GEMRADIO_OLLAMA_MODEL", "gemma4:12b-it-qat")

# Context size decides whether the model reaches the GPU at all.  Ollama sizes
# the KV cache from it, and a large window (the daemon here defaults to 16384
# through OLLAMA_CONTEXT_LENGTH) pushes a 12B model past 8 GB of VRAM, so it
# silently loads on the CPU instead — four times slower.  Every request asks
# for its own num_ctx so the station never inherits that.
OLLAMA_NUM_CTX = int(_env_float("GEMRADIO_NUM_CTX", 8192))
OLLAMA_TIMEOUT = _env_float("GEMRADIO_OLLAMA_TIMEOUT", 180.0)
# How long Ollama keeps the model resident after the last request.  The station
# also unloads explicitly on shutdown so nothing is left holding RAM.
OLLAMA_KEEP_ALIVE = os.environ.get("GEMRADIO_KEEP_ALIVE", "10m")

# --------------------------------------------------------------------------
# Audio engine
# --------------------------------------------------------------------------

SAMPLE_RATE = 48000
CHANNELS = 2
BLOCK_FRAMES = 1024                       # 21.3 ms per mix block
SINK_LATENCY_MS = int(_env_float("GEMRADIO_SINK_LATENCY_MS", 220))

CROSSFADE_SECONDS = _env_float("GEMRADIO_CROSSFADE", 7.0)
DUCK_LEVEL = _env_float("GEMRADIO_DUCK_LEVEL", 0.20)      # ~ -14 dB under speech
DUCK_ATTACK_SECONDS = _env_float("GEMRADIO_DUCK_ATTACK", 0.9)
DUCK_RELEASE_SECONDS = _env_float("GEMRADIO_DUCK_RELEASE", 2.2)
DUCK_HOLD_SECONDS = _env_float("GEMRADIO_DUCK_HOLD", 0.5)
VOICE_GAIN = _env_float("GEMRADIO_VOICE_GAIN", 1.0)
MUSIC_GAIN = _env_float("GEMRADIO_MUSIC_GAIN", 0.82)
MASTER_VOLUME = _env_float("GEMRADIO_VOLUME", 0.85)
DECK_PREBUFFER_SECONDS = 2.0

# --------------------------------------------------------------------------
# Programme direction
# --------------------------------------------------------------------------

MIN_TRACK_SECONDS = _env_float("GEMRADIO_MIN_TRACK", 60.0)
MAX_TRACK_SECONDS = _env_float("GEMRADIO_MAX_TRACK", 900.0)
ARTIST_COOLDOWN = 5          # tracks before the same artist may return
TRACK_HISTORY = 300          # tracks before the same file may return
BLOCK_MIN_TRACKS = 5         # a "show block" keeps one language/identity
BLOCK_MAX_TRACKS = 9
TALK_EVERY_MIN = 2           # DJ speaks at least every N track boundaries
TALK_EVERY_MAX = 3

# How far before the end of a track the DJ starts speaking, on top of the
# measured speech duration.  The next song rises under the closing words.
SPEECH_TAIL_SECONDS = _env_float("GEMRADIO_SPEECH_TAIL", 4.0)
SPEECH_OVERLAP_RATIO = _env_float("GEMRADIO_SPEECH_OVERLAP", 0.72)

STATION_NAME = os.environ.get("GEMRADIO_STATION", "GEMRADIO")
STATION_FREQUENCY = os.environ.get("GEMRADIO_FREQUENCY", "98.6")


# --------------------------------------------------------------------------
# Running profiles
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Profile:
    """How hard the station is allowed to work.

    Low power exists because the models here run on the CPU: it plans a longer
    run of records at once, talks less often, never runs two voices at a time,
    skips the Whisper pass entirely and asks the smallest installed model for
    shorter answers.
    """
    key: str
    label: str
    model: str                    # "" means "smallest model installed"
    talk_every: tuple[int, int]   # speak every N..M track handovers
    allow_duet: bool
    use_whisper: bool
    num_predict: int
    num_ctx: int
    block_tracks: tuple[int, int]  # records per show before a new identity
    llm_show_names: bool
    max_lines: int                # how many spoken lines a link may run to
    words_per_line: int
    total_words: int              # budget for the whole link, all lines together
    max_speech_seconds: float     # hard cap; extra lines are dropped
    description: str


PROFILES: dict[str, Profile] = {
    "full": Profile(
        key="full",
        label="FULL POWER",
        model=OLLAMA_MODEL,       # the bigger model, GPU-accelerated
        talk_every=(TALK_EVERY_MIN, TALK_EVERY_MAX),
        allow_duet=True,
        use_whisper=True,
        num_predict=700,
        num_ctx=OLLAMA_NUM_CTX,
        block_tracks=(BLOCK_MIN_TRACKS, BLOCK_MAX_TRACKS),
        llm_show_names=True,
        max_lines=3,
        words_per_line=26,
        total_words=58,
        max_speech_seconds=26.0,
        description="The larger model on the GPU, longer and funnier links "
                    "about both records, Whisper listening, two hosts.",
    ),
    "low": Profile(
        key="low",
        label="LOW POWER",
        model="",                 # resolved to the smallest installed model
        talk_every=(3, 5),
        allow_duet=False,
        use_whisper=False,
        num_predict=200,
        num_ctx=4096,
        block_tracks=(10, 16),
        llm_show_names=False,
        max_lines=1,
        words_per_line=30,
        total_words=30,
        max_speech_seconds=14.0,
        description="Smallest model, longer runs of music, one voice, "
                    "no Whisper pass.",
    ),
}

DEFAULT_PROFILE = os.environ.get("GEMRADIO_PROFILE", "").strip().lower()
if DEFAULT_PROFILE not in PROFILES:
    DEFAULT_PROFILE = "low" if _env_flag("GEMRADIO_LOW_POWER") else "full"


# --------------------------------------------------------------------------
# Voices
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Voice:
    key: str
    language: str          # "en" | "fr" | "it"
    gender: str            # "F" | "M"
    model: Path
    speaker_id: int | None = None
    length_scale: float = 1.0
    display: str = ""

    @property
    def available(self) -> bool:
        return self.model.is_file()


def _v(rel: str) -> Path:
    return PIPER_VOICES_DIR / rel


VOICES: dict[str, Voice] = {
    "en_F": Voice(
        key="en_F", language="en", gender="F",
        model=_v("en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx"),
        length_scale=1.02, display="Jenny",
    ),
    "en_F2": Voice(
        key="en_F2", language="en", gender="F",
        model=_v("en/en_GB/cori/high/en_GB-cori-high.onnx"),
        length_scale=1.02, display="Cori",
    ),
    "en_F3": Voice(
        key="en_F3", language="en", gender="F",
        model=_v("en/en_GB/alba/medium/en_GB-alba-medium.onnx"),
        length_scale=1.02, display="Alba",
    ),
    "en_M": Voice(
        key="en_M", language="en", gender="M",
        model=_v("en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx"),
        length_scale=1.04, display="Alan",
    ),
    "fr_F": Voice(
        key="fr_F", language="fr", gender="F",
        model=_v("fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx"),
        speaker_id=0, length_scale=1.03, display="Jessica",
    ),
    "fr_M": Voice(
        key="fr_M", language="fr", gender="M",
        model=_v("fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx"),
        speaker_id=1, length_scale=1.05, display="Pierre",
    ),
    "it_F": Voice(
        key="it_F", language="it", gender="F",
        model=_v("it/it_IT/paola/medium/it_IT-paola-medium.onnx"),
        length_scale=1.03, display="Paola",
    ),
}

LANGUAGE_NAMES = {"en": "ENGLISH", "fr": "FRANÇAIS", "it": "ITALIANO"}


def available_voices() -> dict[str, Voice]:
    return {k: v for k, v in VOICES.items() if v.available}


def voices_for(language: str) -> list[Voice]:
    return [v for v in available_voices().values() if v.language == language]


def available_languages() -> list[str]:
    langs = []
    for lang in ("en", "fr", "it"):
        if voices_for(lang):
            langs.append(lang)
    return langs or ["en"]


def pick_voice(language: str, gender: str) -> Voice | None:
    """Best voice for a language/gender, falling back within the language."""
    pool = voices_for(language)
    if not pool:
        return None
    exact = [v for v in pool if v.gender == gender]
    return (exact or pool)[0]


def ensure_dirs() -> None:
    for d in (STATE_DIR, CACHE_DIR, COVER_DIR, TTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
