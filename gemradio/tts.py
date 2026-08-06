"""Speech synthesis for the DJ.

Talks to a long-lived Piper worker process and returns audio already converted
to the mixer's format (48 kHz stereo float32), with a short lead-in/lead-out so
speech never starts flush against the duck.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .library import has_unspeakable_script, strip_unspeakable
from .audio import CH, SR, VoiceItem
from .logging_util import get_logger

log = get_logger(__name__)


@dataclass
class Utterance:
    voice_key: str
    text: str
    audio: np.ndarray
    duration: float

    @property
    def voice(self):
        return config.VOICES.get(self.voice_key)


class TTSError(RuntimeError):
    pass


# Piper reads the bare station name as letters; spacing it makes every voice
# say "Gem Radio" the way a real announcer would.
_SPOKEN_FIXES = (
    (config.STATION_NAME, "Gem Radio"),
    (config.STATION_NAME.title(), "Gem Radio"),
    ("GemRadio", "Gem Radio"),
)


def spoken_form(text: str, language: str = "") -> str:
    for src, dst in _SPOKEN_FIXES:
        if src:
            text = text.replace(src, dst)
    # Last line of defence: a Hebrew or Cyrillic run reaching a Latin-script
    # voice comes out as noise, so drop it even if the model slipped one in.
    if language in ("en", "fr", "it") and has_unspeakable_script(text):
        cleaned = strip_unspeakable(text)
        if len(cleaned.split()) >= 3:
            log.info("stripped an unpronounceable run from: %.70s", text)
            text = cleaned
    return text


class PiperTTS:
    """Client for the Piper worker subprocess."""

    # Speech is essentially never repeated, so the cache exists only to survive
    # restarts and repeated station IDs.  Left alone it would grow without
    # bound over days of broadcasting, so it is trimmed on every start.
    CACHE_BUDGET_BYTES = 150 * 1024 * 1024

    def __init__(self):
        config.ensure_dirs()
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._counter = 0
        self.ready = False
        self._trim_cache()

    def _trim_cache(self) -> None:
        try:
            files = [(p.stat().st_mtime, p.stat().st_size, p)
                     for p in config.TTS_DIR.iterdir() if p.is_file()]
        except OSError:
            return
        total = sum(size for _, size, _ in files)
        if total <= self.CACHE_BUDGET_BYTES:
            return
        files.sort()                       # oldest first
        removed = 0
        for _, size, path in files:
            if total <= self.CACHE_BUDGET_BYTES * 0.7:
                break
            try:
                path.unlink()
                total -= size
                removed += 1
            except OSError:
                pass
        log.info("tts cache trimmed: %d files removed", removed)

    # -- process management -------------------------------------------------
    def _spawn(self) -> None:
        env = dict(os.environ)
        site = str(config.PIPER_SITE)
        if Path(site).is_dir():
            env["PYTHONPATH"] = site + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.setdefault("OMP_NUM_THREADS", "4")

        worker = Path(__file__).with_name("piper_worker.py")
        log.info("starting piper worker (site=%s)", site)
        self._proc = subprocess.Popen(
            [sys.executable, "-u", str(worker)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL if not os.environ.get("GEMRADIO_DEBUG") else None,
            env=env, text=True, bufsize=1,
        )
        line = self._proc.stdout.readline()
        if not line:
            raise TTSError("piper worker failed to start")
        self.ready = True

    def _ensure(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self.ready = False
            self._spawn()

    def close(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=3)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
            self._proc = None
            self.ready = False

    # -- synthesis ----------------------------------------------------------
    def _request(self, payload: dict, timeout: float = 180.0) -> dict:
        with self._lock:
            self._ensure()
            self._counter += 1
            payload["id"] = self._counter
            assert self._proc and self._proc.stdin and self._proc.stdout
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                line = self._proc.stdout.readline()
                if not line:
                    raise TTSError("piper worker died")
                try:
                    resp = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if resp.get("id") == payload["id"]:
                    return resp
            raise TTSError("piper worker timed out")

    def synthesize(self, text: str, voice_key: str,
                   length_scale: float | None = None) -> Utterance | None:
        voice = config.VOICES.get(voice_key)
        if voice is None or not voice.available:
            log.warning("voice %s unavailable", voice_key)
            return None
        text = (text or "").strip()
        if not text:
            return None
        text = spoken_form(text, voice.language)

        scale = length_scale if length_scale is not None else voice.length_scale
        digest = hashlib.sha1(
            f"{voice_key}|{scale:.3f}|{text}".encode("utf-8", "replace")
        ).hexdigest()[:24]
        # Only Piper's own 22 kHz mono output is cached; the 48 kHz stereo
        # version is nine times larger and ffmpeg regenerates it in milliseconds.
        wav_path = config.TTS_DIR / f"{digest}.wav"

        if not wav_path.is_file():
            resp = self._request({
                "model": str(voice.model),
                "speaker_id": voice.speaker_id,
                "length_scale": scale,
                "text": text,
                "out": str(wav_path),
                "use_cuda": config.PIPER_USE_CUDA,
            })
            if not resp.get("ok"):
                log.error("piper failed: %s", resp.get("error"))
                return None

        audio = self._to_mix_format(wav_path)
        if audio is None:
            return None
        return Utterance(voice_key, text, audio, audio.shape[0] / SR)

    @staticmethod
    def _to_mix_format(wav_path: Path) -> np.ndarray | None:
        """Resample Piper's 22.05 kHz mono to the mixer's 48 kHz stereo."""
        cmd = [
            config.FFMPEG, "-v", "quiet", "-nostdin", "-i", str(wav_path),
            "-af", "loudnorm=I=-17:TP=-2:LRA=9",
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", str(SR), "-ac", str(CH), "-",
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, timeout=120).stdout
        except Exception:
            log.exception("ffmpeg conversion failed")
            return None
        if not out:
            return None
        arr = np.frombuffer(out, dtype="<f4")
        usable = (arr.size // CH) * CH
        audio = arr[:usable].reshape(-1, CH).astype(np.float32, copy=True)

        # Small lead-in/out silence plus fades so the duck envelope has room.
        lead = int(0.18 * SR)
        tail = int(0.30 * SR)
        pad = np.zeros((lead + audio.shape[0] + tail, CH), dtype=np.float32)
        pad[lead:lead + audio.shape[0]] = audio
        ramp = int(0.02 * SR)
        if pad.shape[0] > 2 * ramp:
            pad[lead:lead + ramp] *= np.linspace(0, 1, ramp, dtype=np.float32)[:, None]
            end = lead + audio.shape[0]
            pad[end - ramp:end] *= np.linspace(1, 0, ramp, dtype=np.float32)[:, None]
        return pad

    # -- convenience --------------------------------------------------------
    def make_items(self, lines: list[tuple[str, str]], gap: float = 0.32,
                   max_seconds: float = 0.0) -> tuple[list[VoiceItem], float]:
        """Synthesize (voice_key, text) pairs into mixer items.

        Returns the items and their total wall-clock duration including gaps.
        A link that would run past `max_seconds` is cut short rather than
        allowed to overrun the end of the record it plays over.
        """
        items: list[VoiceItem] = []
        for idx, (voice_key, text) in enumerate(lines):
            utt = self.synthesize(text, voice_key)
            if utt is None:
                continue
            items.append(VoiceItem(
                audio=utt.audio,
                gap_after=gap,
                meta={"text": text, "voice": voice_key,
                      "display": (utt.voice.display if utt.voice else voice_key),
                      "index": idx, "last": False},
            ))

        if max_seconds:
            items = self._fit_to_budget(items, max_seconds, gap)
        return items, self._finalize(items)

    @staticmethod
    def _fit_to_budget(items: list[VoiceItem], max_seconds: float,
                       gap: float) -> list[VoiceItem]:
        """Drop lines from the middle until the link fits the budget.

        The last line is the one that announces the record coming next and the
        first sets up the handover, so when something has to go it is taken
        from between them rather than off the end.
        """
        def duration(seq: list[VoiceItem]) -> float:
            return sum(i.audio.shape[0] / SR for i in seq) + gap * max(0, len(seq) - 1)

        while len(items) > 1 and duration(items) > max_seconds:
            drop = 1 if len(items) > 2 else 0
            dropped = items.pop(drop)
            log.info("link over the %.0fs cap: dropped line %d (%.60s…)",
                     max_seconds, dropped.meta.get("index", drop),
                     dropped.meta.get("text", ""))
        return items

    @staticmethod
    def _finalize(items: list[VoiceItem]) -> float:
        """Close the last line's gap and return the link's true duration."""
        if not items:
            return 0.0
        items[-1].gap_after = 0.0
        items[-1].meta["last"] = True
        return sum(i.audio.shape[0] / SR + i.gap_after for i in items)

    def warm_up(self, voice_keys: list[str]) -> None:
        """Pre-load models so the first on-air line is not delayed."""
        for key in voice_keys:
            voice = config.VOICES.get(key)
            if voice and voice.available:
                try:
                    self.synthesize("GemRadio.", key)
                except Exception:
                    log.debug("warm-up failed for %s", key, exc_info=True)
