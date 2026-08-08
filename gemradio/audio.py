"""The GemRadio audio engine.

A single mixer thread runs the whole station output:

  * two music decks, each fed by an ffmpeg decoder thread,
  * equal-power crossfades between them,
  * a voice deck for the DJ that ducks the music underneath it,
  * one PCM sink (paplay / pw-play / aplay) which also acts as the clock,
    because writing to it blocks at exactly real-time speed.

Everything is float32 at 48 kHz stereo internally and converted to s16le only
on the way out.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import config
from .logging_util import get_logger

log = get_logger(__name__)

SR = config.SAMPLE_RATE
CH = config.CHANNELS
BLOCK = config.BLOCK_FRAMES
BLOCK_SECONDS = BLOCK / SR
BYTES_PER_FRAME = CH * 4  # float32 stereo


# --------------------------------------------------------------------------
# Decoding deck
# --------------------------------------------------------------------------

class Deck:
    """One music source: ffmpeg decodes into a bounded queue of mix blocks."""

    def __init__(self, path: str, duration: float = 0.0, start_at: float = 0.0):
        self.path = path
        self.duration = duration
        self.start_at = start_at
        self.frames_played = 0
        self.gain = 1.0
        self.target_gain = 1.0
        self.fade_dir = 0            # -1 fading out, +1 fading in, 0 steady
        self.fade_pos = 0.0
        self.fade_len = 0.0
        self.finished = False        # decoder exhausted and buffer drained
        self.started = False

        self._queue: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._space = threading.Condition(self._lock)
        self._eof = False
        self._stop = threading.Event()
        self._max_blocks = max(8, int(config.DECK_PREBUFFER_SECONDS / BLOCK_SECONDS))

        cmd = [config.FFMPEG, "-v", "quiet", "-nostdin"]
        if start_at > 0.05:
            cmd += ["-ss", f"{start_at:.3f}"]
        cmd += [
            "-i", path,
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", str(SR), "-ac", str(CH),
            "-",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
        )
        self._thread = threading.Thread(target=self._reader, name="deck-reader", daemon=True)
        self._thread.start()

    # -- decoder thread -----------------------------------------------------
    def _reader(self) -> None:
        want = BLOCK * BYTES_PER_FRAME
        stdout = self._proc.stdout
        try:
            while not self._stop.is_set():
                buf = bytearray()
                while len(buf) < want:
                    chunk = stdout.read(want - len(buf))
                    if not chunk:
                        break
                    buf.extend(chunk)
                if not buf:
                    break
                arr = np.frombuffer(bytes(buf), dtype="<f4")
                usable = (arr.size // CH) * CH
                block = arr[:usable].reshape(-1, CH).astype(np.float32, copy=True)
                if block.shape[0] < BLOCK:
                    pad = np.zeros((BLOCK - block.shape[0], CH), dtype=np.float32)
                    block = np.vstack([block, pad])
                with self._space:
                    while len(self._queue) >= self._max_blocks and not self._stop.is_set():
                        self._space.wait(0.2)
                    if self._stop.is_set():
                        break
                    self._queue.append(block)
                if len(buf) < want:
                    break
        except Exception:
            log.debug("deck reader ended for %s", self.path, exc_info=True)
        finally:
            with self._space:
                self._eof = True
                self._space.notify_all()
            try:
                if stdout:
                    stdout.close()
            except Exception:
                pass

    @property
    def buffered_blocks(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def ready(self) -> bool:
        """Enough audio buffered (or EOF) that playback will not stutter."""
        with self._lock:
            return self._eof or len(self._queue) >= min(self._max_blocks, 24)

    def read(self) -> np.ndarray | None:
        """Next mix block, or None when the deck is done."""
        with self._space:
            if self._queue:
                block = self._queue.popleft()
                self._space.notify_all()
                self.started = True
                self.frames_played += BLOCK
                return block
            if self._eof:
                self.finished = True
                return None
        # Decoder is alive but starved: emit silence rather than glitching.
        self.frames_played += BLOCK
        return np.zeros((BLOCK, CH), dtype=np.float32)

    # -- fades --------------------------------------------------------------
    def fade_in(self, seconds: float) -> None:
        self.gain = 0.0
        self.fade_dir = 1
        self.fade_pos = 0.0
        self.fade_len = max(0.01, seconds)

    def fade_out(self, seconds: float) -> None:
        self.fade_dir = -1
        self.fade_pos = 0.0
        self.fade_len = max(0.01, seconds)
        self._fade_from = self.gain

    def advance_fade(self) -> None:
        if not self.fade_dir:
            return
        self.fade_pos += BLOCK_SECONDS
        p = min(1.0, self.fade_pos / self.fade_len)
        if self.fade_dir > 0:
            self.gain = math.sin(p * math.pi / 2)          # equal power in
        else:
            self.gain = getattr(self, "_fade_from", 1.0) * math.cos(p * math.pi / 2)
        if p >= 1.0:
            self.fade_dir = 0
            self.gain = 1.0 if self.gain > 0.5 else 0.0

    @property
    def position(self) -> float:
        return self.start_at + self.frames_played / SR

    @property
    def remaining(self) -> float:
        if self.duration <= 0:
            return 1e9
        return max(0.0, self.duration - self.position)

    def close(self) -> None:
        """Tear the deck down without ever blocking the caller.

        close() runs on the mixer thread when a faded-out deck is dropped, so
        waiting on ffmpeg here would stall the whole output stream; the reap is
        handed to a throwaway thread instead.
        """
        self._stop.set()
        with self._space:
            self._queue.clear()
            self._space.notify_all()
        try:
            if self._proc.poll() is None:
                self._proc.kill()
        except Exception:
            pass
        threading.Thread(target=self._reap, name="deck-reap", daemon=True).start()

    def _reap(self) -> None:
        try:
            self._proc.wait(timeout=5)
        except Exception:
            pass


# --------------------------------------------------------------------------
# Voice queue
# --------------------------------------------------------------------------

@dataclass
class VoiceItem:
    audio: np.ndarray             # float32 (n, 2) at 48 kHz
    gap_after: float = 0.35
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Engine state snapshot
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Output devices
# --------------------------------------------------------------------------

STREAM_NAME = "GemRadio"


def list_output_devices() -> list[tuple[str, str]]:
    """Available sinks as (name, description), best-effort.

    An empty name means "whatever the system default is", which is the entry
    the station starts on.
    """
    devices: list[tuple[str, str]] = [("", "System default")]
    if not shutil.which("pactl"):
        return devices
    try:
        out = subprocess.run(["pactl", "list", "sinks"], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:
        return devices
    name = ""
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Description:") and name:
            devices.append((name, line.split(":", 1)[1].strip()))
            name = ""
    return devices


def default_sink() -> str:
    try:
        return subprocess.run(["pactl", "get-default-sink"], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def _find_our_sink_input() -> str:
    """Index of the station's own playback stream, by its media name."""
    try:
        out = subprocess.run(["pactl", "list", "sink-inputs"], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:
        return ""
    index = ""
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Sink Input #"):
            index = stripped.split("#", 1)[1].strip()
        elif f'media.name = "{STREAM_NAME}"' in stripped and index:
            return index
    return ""


@dataclass
class EngineState:
    running: bool = False
    device: str = ""
    track_path: str = ""
    position: float = 0.0
    duration: float = 0.0
    remaining: float = 0.0
    speaking: bool = False
    crossfading: bool = False
    duck: float = 1.0
    vu_left: float = 0.0
    vu_right: float = 0.0
    volume: float = config.MASTER_VOLUME
    voice_meta: dict = field(default_factory=dict)


class AudioEngine:
    """Mixes music and speech and pushes the result to the system audio sink."""

    def __init__(self, on_event: Callable[[str, dict], None] | None = None,
                 device: str = ""):
        self.on_event = on_event
        self._device = device
        self._lock = threading.RLock()
        self._decks: list[Deck] = []
        self._voice: deque[VoiceItem] = deque()
        self._voice_cur: VoiceItem | None = None
        self._voice_pos = 0
        self._voice_gap = 0.0
        self._voice_meta: dict = {}

        self._duck = 1.0
        self._duck_hold = 0.0
        self._volume = float(config.MASTER_VOLUME)
        self._volume_smoothed = float(config.MASTER_VOLUME)
        self._vu = (0.0, 0.0)

        self._running = False
        self._thread: threading.Thread | None = None
        self._sink: subprocess.Popen | None = None
        self._paused = False
        self._sink_cmd = self._choose_sink(device)

    # -- sink ---------------------------------------------------------------
    @staticmethod
    def _choose_sink(device: str = "") -> list[str]:
        lat = str(config.SINK_LATENCY_MS)
        if shutil.which("paplay"):
            cmd = ["paplay", "--raw", "--format=s16le", f"--rate={SR}",
                   f"--channels={CH}", f"--latency-msec={lat}",
                   f"--stream-name={STREAM_NAME}"]
            if device:
                cmd.append(f"--device={device}")
            return cmd
        if shutil.which("pw-play"):
            cmd = ["pw-play", "--format=s16", f"--rate={SR}", f"--channels={CH}"]
            if device:
                cmd.append(f"--target={device}")
            return cmd + ["-"]
        if shutil.which("aplay"):
            return ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(SR),
                    "-c", str(CH), "-"]
        raise RuntimeError("no audio sink found (need paplay, pw-play or aplay)")

    @staticmethod
    def device_selection_supported() -> bool:
        return bool(shutil.which("paplay") or shutil.which("pw-play"))

    def _open_sink(self) -> None:
        self._sink_cmd = self._choose_sink(self._device)
        log.info("audio sink: %s", " ".join(self._sink_cmd))
        self._sink = subprocess.Popen(
            self._sink_cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    @property
    def device(self) -> str:
        return self._device

    def set_device(self, name: str) -> bool:
        """Send the station's audio to a particular output, and only ours.

        Preferred route is asking PulseAudio to move our existing stream, which
        is seamless and leaves every other application where it is.  If that is
        not possible the sink process is reopened instead, which costs a short
        gap but has the same effect.
        """
        name = name or ""
        with self._lock:
            self._device = name
            running = self._running and self._sink is not None

        target = name or default_sink()
        if running and target and shutil.which("pactl"):
            index = _find_our_sink_input()
            if index:
                try:
                    done = subprocess.run(
                        ["pactl", "move-sink-input", index, target],
                        capture_output=True, timeout=5)
                    if done.returncode == 0:
                        log.info("output moved to %s", target)
                        return True
                    log.debug("move-sink-input failed: %s",
                              done.stderr.decode(errors="replace").strip())
                except Exception:
                    log.debug("move-sink-input raised", exc_info=True)

        if running:
            return self._reopen_sink()
        return True

    def _reopen_sink(self) -> bool:
        """Swap the sink process under the mixer, keeping the stream going."""
        old = self._sink
        try:
            new_cmd = self._choose_sink(self._device)
            new = subprocess.Popen(new_cmd, stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        except Exception:
            log.exception("could not open the new output")
            return False
        with self._lock:
            self._sink_cmd = new_cmd
            self._sink = new
        if old:
            try:
                if old.stdin:
                    old.stdin.close()
            except Exception:
                pass
            threading.Thread(target=self._close_process, args=(old,),
                             daemon=True).start()
        log.info("output reopened on %s", self._device or "system default")
        return True

    @staticmethod
    def _close_process(proc: subprocess.Popen) -> None:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        self._open_sink()
        self._thread = threading.Thread(target=self._run, name="mixer", daemon=True)
        self._thread.start()
        log.info("audio engine started")

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        with self._lock:
            for d in self._decks:
                d.close()
            self._decks.clear()
            self._voice.clear()
            self._voice_cur = None
        if self._sink:
            try:
                if self._sink.stdin:
                    self._sink.stdin.close()
            except Exception:
                pass
            try:
                self._sink.terminate()
                self._sink.wait(timeout=2)
            except Exception:
                try:
                    self._sink.kill()
                except Exception:
                    pass
            self._sink = None
        log.info("audio engine stopped")

    # -- transport ----------------------------------------------------------
    def play(self, path: str, duration: float = 0.0, fade_in: float = 0.0,
             meta: dict | None = None) -> None:
        """Start a track immediately, replacing anything currently playing."""
        deck = Deck(path, duration)
        with self._lock:
            for d in self._decks:
                d.close()
            self._decks = [deck]
            if fade_in > 0:
                deck.fade_in(fade_in)
        self._emit("track_started", {"path": path, "duration": duration, **(meta or {})})

    def crossfade_to(self, path: str, duration: float = 0.0,
                     seconds: float | None = None, meta: dict | None = None) -> None:
        """Bring a new track up while the current one fades away."""
        seconds = config.CROSSFADE_SECONDS if seconds is None else seconds
        deck = Deck(path, duration)
        # Let the decoder fill before the fade starts so the entry is clean.
        deadline = time.monotonic() + 4.0
        while not deck.ready and time.monotonic() < deadline:
            time.sleep(0.02)
        with self._lock:
            for d in self._decks:
                if not d.finished:
                    d.fade_out(seconds)
            deck.fade_in(seconds)
            self._decks.append(deck)
            if len(self._decks) > 3:
                old = self._decks.pop(0)
                old.close()
        self._emit("track_started", {"path": path, "duration": duration, **(meta or {})})

    def fade_out_all(self, seconds: float = 2.5) -> None:
        with self._lock:
            for d in self._decks:
                d.fade_out(seconds)

    def clear_music(self) -> None:
        with self._lock:
            for d in self._decks:
                d.close()
            self._decks.clear()

    # -- speech -------------------------------------------------------------
    def speak(self, items: list[VoiceItem]) -> None:
        with self._lock:
            for it in items:
                self._voice.append(it)

    def speaking(self) -> bool:
        with self._lock:
            return self._voice_cur is not None or bool(self._voice) or self._voice_gap > 0

    def cancel_speech(self) -> None:
        with self._lock:
            self._voice.clear()
            self._voice_cur = None
            self._voice_pos = 0
            self._voice_gap = 0.0

    # -- controls -----------------------------------------------------------
    def set_volume(self, value: float) -> None:
        with self._lock:
            self._volume = max(0.0, min(1.0, float(value)))

    @property
    def volume(self) -> float:
        return self._volume

    def state(self) -> EngineState:
        with self._lock:
            # The most recently started deck is the one the station considers
            # current, so during a crossfade the panel already shows the
            # incoming record's position rather than the outgoing one's.
            primary = next((d for d in reversed(self._decks) if not d.finished), None)
            crossfading = sum(1 for d in self._decks if not d.finished and d.gain > 0.02) > 1
            return EngineState(
                running=self._running,
                device=self._device,
                track_path=primary.path if primary else "",
                position=primary.position if primary else 0.0,
                duration=primary.duration if primary else 0.0,
                remaining=primary.remaining if primary else 0.0,
                speaking=self._voice_cur is not None or bool(self._voice),
                crossfading=crossfading,
                duck=self._duck,
                vu_left=self._vu[0], vu_right=self._vu[1],
                volume=self._volume,
                voice_meta=dict(self._voice_meta),
            )

    def deck_for(self, path: str) -> Deck | None:
        with self._lock:
            for d in self._decks:
                if d.path == path and not d.finished:
                    return d
        return None

    # -- mixing -------------------------------------------------------------
    def _emit(self, kind: str, payload: dict) -> None:
        if self.on_event:
            try:
                self.on_event(kind, payload)
            except Exception:
                log.exception("event handler failed for %s", kind)

    def _mix_music(self, out: np.ndarray) -> None:
        dead: list[Deck] = []
        for deck in list(self._decks):
            block = deck.read()
            if block is None:
                dead.append(deck)
                continue
            deck.advance_fade()
            if deck.gain > 0.0005:
                out += block * deck.gain
            elif deck.fade_dir == 0 and deck.started:
                # Faded fully out and no longer needed.
                dead.append(deck)
        for deck in dead:
            deck.close()
            if deck in self._decks:
                self._decks.remove(deck)
            self._emit("track_finished", {"path": deck.path})

    def _mix_voice(self, out: np.ndarray) -> bool:
        """Add DJ speech on top; returns True when speech is sounding."""
        written = 0
        active = False
        while written < BLOCK:
            if self._voice_gap > 0:
                skip = min(BLOCK - written, int(self._voice_gap * SR))
                skip = max(skip, 1)
                self._voice_gap -= skip / SR
                written += skip
                continue
            if self._voice_cur is None:
                if not self._voice:
                    break
                self._voice_cur = self._voice.popleft()
                self._voice_pos = 0
                self._voice_meta = dict(self._voice_cur.meta)
                self._emit("speech_started", dict(self._voice_cur.meta))
            audio = self._voice_cur.audio
            take = min(BLOCK - written, audio.shape[0] - self._voice_pos)
            if take > 0:
                out[written:written + take] += (
                    audio[self._voice_pos:self._voice_pos + take] * config.VOICE_GAIN
                )
                self._voice_pos += take
                written += take
                active = True
            if self._voice_pos >= audio.shape[0]:
                gap = self._voice_cur.gap_after
                meta = self._voice_cur.meta
                self._voice_cur = None
                self._voice_pos = 0
                self._voice_gap = gap
                self._emit("speech_segment_done", dict(meta))
                if not self._voice and gap <= 0.001:
                    self._voice_meta = {}
                    self._emit("speech_finished", {})
        return active

    def _run(self) -> None:
        silence = np.zeros((BLOCK, CH), dtype=np.float32)
        duck_target_prev = 1.0
        was_speaking = False
        while True:
            with self._lock:
                if not self._running:
                    break

                music = silence.copy()
                self._mix_music(music)

                voice = np.zeros((BLOCK, CH), dtype=np.float32)
                voice_active = self._mix_voice(voice)
                speaking_now = voice_active or self._voice_cur is not None or bool(self._voice)

                if voice_active:
                    self._duck_hold = config.DUCK_HOLD_SECONDS
                elif self._duck_hold > 0:
                    self._duck_hold -= BLOCK_SECONDS

                target = config.DUCK_LEVEL if (voice_active or self._duck_hold > 0) else 1.0
                tau = (config.DUCK_ATTACK_SECONDS if target < self._duck
                       else config.DUCK_RELEASE_SECONDS)
                alpha = 1.0 - math.exp(-BLOCK_SECONDS / max(0.05, tau))
                self._duck += (target - self._duck) * alpha
                duck_target_prev = target

                if was_speaking and not speaking_now:
                    self._voice_meta = {}
                    self._emit("speech_finished", {})
                was_speaking = speaking_now

                self._volume_smoothed += (self._volume - self._volume_smoothed) * 0.05
                vol = self._volume_smoothed

                mixed = (music * (config.MUSIC_GAIN * self._duck) + voice) * vol
                # Gentle soft clip keeps speech-over-music peaks from crackling.
                np.tanh(mixed * 1.1, out=mixed)
                mixed *= 0.92

                peak = np.abs(mixed).max(axis=0) if mixed.size else np.zeros(2)
                self._vu = (float(peak[0]), float(peak[1]))
                sink = self._sink

            pcm = np.clip(mixed, -1.0, 1.0)
            data = (pcm * 32767.0).astype("<i2").tobytes()
            try:
                if sink and sink.stdin:
                    sink.stdin.write(data)      # blocks at real-time rate
                else:
                    time.sleep(BLOCK_SECONDS)
            except (BrokenPipeError, ValueError):
                log.error("audio sink closed unexpectedly")
                break
        log.debug("mixer thread exited")
