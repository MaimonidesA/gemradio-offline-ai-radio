"""The station: everything running together, on a clock.

The timing is the whole point.  For each handover the director prepares the
next record while the current one plays, Whisper listens to it, Gemma writes
the link, Piper voices it — and only then does the control loop schedule it:

    speech starts at  crossfade + overlap * speech_length  before the end,
    the crossfade begins once the DJ is ~70% through the line,

so the next record rises underneath the closing words and the outgoing one is
gone by the time they finish.  That is what makes it sound like radio instead
of a playlist with announcements.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config, lyrics
from .audio import AudioEngine, VoiceItem
from .director import Director, Script, Show
from .library import Library, Track
from .llm import Ollama
from .logging_util import get_logger
from .tts import PiperTTS

log = get_logger(__name__)


@dataclass
class Segment:
    """A prepared upcoming record, with its DJ link if there is one."""
    track: Track
    script: Script | None = None
    items: list[VoiceItem] = field(default_factory=list)
    speech_seconds: float = 0.0
    ready: threading.Event = field(default_factory=threading.Event)
    kind: str = "transition"
    # Set once the control loop has put this segment on air.  Preparation that
    # is still running for it must then stop rather than publish stale work.
    consumed: bool = False

    @property
    def has_speech(self) -> bool:
        return bool(self.items)


class Station:
    PHASE_IDLE = "idle"
    PHASE_PLAYING = "playing"
    PHASE_TALKING = "talking"
    PHASE_HANDOVER = "handover"

    def __init__(self, library: Library | None = None, profile: str | None = None):
        config.ensure_dirs()
        self.library = library or Library()
        self.brain = Ollama()
        self.profile = config.PROFILES[profile or config.DEFAULT_PROFILE]
        self.director = Director(self.library, self.brain, self.profile)
        self.tts = PiperTTS()
        self.engine = AudioEngine(on_event=self._on_engine_event)

        self._lock = threading.RLock()
        self._running = False
        self._control: threading.Thread | None = None
        self._prep: threading.Thread | None = None
        self._prep_wake = threading.Event()

        self.current: Track | None = None
        self.current_started = 0.0
        self.pending: Segment | None = None
        self.show: Show | None = None
        self.phase = self.PHASE_IDLE
        self.status = "Ready"
        self.on_air_text = ""
        self.on_air_voice = ""
        self.last_script: Script | None = None
        self._crossfade_at = 0.0
        self._last_station_id_hour = -1
        self._tracks_since_id = 0
        self._segments_prepared = 0
        self._dead_air_since = 0.0
        self._shutdown_done = False
        self._speech_log: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def set_profile(self, key: str) -> None:
        """Switch between full and low-power running, while on air.

        Only work that has not been prepared yet is affected: whatever is
        already synthesized still goes out, so the switch is never audible as
        a glitch.
        """
        profile = config.PROFILES.get(key)
        if profile is None or profile is self.profile:
            return
        with self._lock:
            self.profile = profile
            self.director.profile = profile
        self._apply_profile_model()
        log.info("profile: %s (%s)", profile.label, self.brain.model)

    def _apply_profile_model(self) -> None:
        """Point the brain at this profile's model, resolving 'smallest'."""
        wanted = self.profile.model or self.brain.smallest_model()
        previous = self.brain.model
        self.brain.set_model(wanted)
        if previous and previous != self.brain.model:
            # Let the daemon release the model we are no longer using.
            was = self.brain.model
            self.brain.model = previous
            self.brain.unload()
            self.brain.model = was

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self.status = "Warming up…"
        self.brain.check()
        self._apply_profile_model()
        self.engine.start()

        self._prep = threading.Thread(target=self._prep_loop, name="prep", daemon=True)
        self._prep.start()
        self._control = threading.Thread(target=self._control_loop, name="control", daemon=True)
        self._control.start()
        log.info("station on air")

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            self.phase = self.PHASE_IDLE
            self.status = "Off air"
        self._prep_wake.set()
        self.engine.cancel_speech()
        self.engine.fade_out_all(1.6)
        time.sleep(1.7)
        self.engine.stop()
        if self._control:
            self._control.join(timeout=3)
        if self._prep:
            self._prep.join(timeout=5)
        with self._lock:
            self.current = None
            self.pending = None
            self.on_air_text = ""
            self._segments_prepared = 0
            self._dead_air_since = 0.0
        log.info("station off air")

    def shutdown(self) -> None:
        """Leave the machine exactly as we found it.

        Stops the mixer and its ffmpeg decoders, ends the Piper worker, halts
        any library scan, and asks Ollama to drop the model rather than letting
        it sit on gigabytes of RAM for its keep-alive window.

        Called from the window's close event, from Qt's aboutToQuit, and from
        an atexit hook, so it has to be safe to run more than once.
        """
        with self._lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
        log.info("shutting down")
        try:
            self.stop()
        finally:
            try:
                self.library.stop_scan()
            except Exception:
                log.debug("scan stop failed", exc_info=True)
            try:
                self.tts.close()
            except Exception:
                log.debug("tts close failed", exc_info=True)
            try:
                if self.brain.available:
                    self.brain.unload()
            except Exception:
                log.debug("model unload failed", exc_info=True)
            log.info("shutdown complete")

    @property
    def running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # preparation (slow work, always ahead of air time)
    # ------------------------------------------------------------------
    def _prep_loop(self) -> None:
        while self._running:
            try:
                with self._lock:
                    idle = self.pending is None
                if idle:
                    if not self._prepare_segment():
                        self.status = "Waiting for the library to fill…"
                        time.sleep(2.0)
                else:
                    self._prep_wake.wait(0.4)
                    self._prep_wake.clear()
            except Exception:
                log.exception("preparation failed")
                time.sleep(1.5)

    def _prepare_segment(self) -> bool:
        """Build the next segment, publishing it as soon as the track is known.

        The segment is published before the slow work (Whisper, Gemma, Piper)
        so the panel can show what is coming.  The control loop may take it on
        air early — with a plain crossfade and no link — in which case the rest
        of the preparation is discarded rather than published twice.
        """
        track = self.director.next_track()
        if track is None:
            return False

        upcoming = self.director.peek_upcoming(5)
        show = self.director.tick_show([track] + upcoming)

        seg = Segment(track=track)
        with self._lock:
            self.show = show
            self.pending = seg
            # Decided here, in the single preparation thread, so it cannot race
            # with the control loop setting `current`.
            if self._segments_prepared == 0:
                seg.kind = "open"
            elif self._should_station_id():
                seg.kind = "station_id"
            self._segments_prepared += 1
            previous = self.current

        speak = seg.kind != "transition" or self.director.should_talk()
        if speak:
            if self.profile.use_whisper:
                self._listen_to(track)
            if not seg.consumed and self._running:
                script = self.director.build_script(seg.kind, previous, track, show)
                if script.lines and not seg.consumed:
                    items, total = self.tts.make_items(script.lines)
                    seg.script = script
                    seg.items = items
                    seg.speech_seconds = total
                    log.info("link ready (%s, %s, %.1fs): %s",
                             seg.kind, script.source, total, script.text[:120])
        if seg.consumed:
            log.debug("segment for %s went on air before its link was ready",
                      track.display)
        seg.ready.set()
        return True

    def _listen_to(self, track: Track) -> None:
        """Let Whisper hear the upcoming record once, then cache the result."""
        if not lyrics.available() or track.lyric_lang:
            return
        fresh = self.library.get(track.path)
        if fresh and fresh.lyric_lang:
            track.lyric, track.lyric_lang = fresh.lyric, fresh.lyric_lang
            return
        try:
            text, lang = lyrics.transcribe(track.path, track.duration)
        except Exception:
            log.debug("whisper failed for %s", track.path, exc_info=True)
            return
        track.lyric, track.lyric_lang = text, (lang or "?")
        self.library.store_lyric(track.path, text, lang or "?")
        if text:
            log.info("whisper[%s]: %.90s", lang, text)

    def _should_station_id(self) -> bool:
        hour = dt.datetime.now().hour
        if hour != self._last_station_id_hour and self._tracks_since_id >= 2:
            self._last_station_id_hour = hour
            self._tracks_since_id = 0
            return True
        return False

    # ------------------------------------------------------------------
    # control loop (the clock)
    # ------------------------------------------------------------------
    def _control_loop(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception:
                log.exception("control loop error")
            time.sleep(0.1)

    def _tick(self) -> None:
        state = self.engine.state()

        if self.current is None:
            self._go_on_air()
            return

        if self._recover_dead_air(state):
            return

        if self.phase == self.PHASE_PLAYING:
            seg = self.pending
            if seg is None or not seg.ready.is_set():
                # Nothing prepared yet: fall back to a clean crossfade so the
                # station never hits silence waiting on the model.
                if state.remaining <= config.CROSSFADE_SECONDS and seg is not None:
                    self._handover(seg, silent=True)
                return

            trigger = self._speech_trigger(seg)
            if state.remaining <= trigger:
                if seg.has_speech:
                    self._start_link(seg)
                else:
                    self._handover(seg, silent=True)

        elif self.phase == self.PHASE_TALKING:
            if time.monotonic() >= self._crossfade_at:
                seg = self.pending
                if seg is not None:
                    self._handover(seg, silent=False)

        elif self.phase == self.PHASE_HANDOVER:
            if not state.speaking and state.remaining > config.CROSSFADE_SECONDS:
                self.phase = self.PHASE_PLAYING

    def _recover_dead_air(self, state) -> bool:
        """Never sit in silence: if the music stopped, start the next record.

        This only fires when something upstream went wrong — a file that failed
        to decode, or a track that ran out before its link was ready.
        """
        if state.track_path or state.speaking:
            self._dead_air_since = 0.0
            return False
        now = time.monotonic()
        if self._dead_air_since == 0.0:
            self._dead_air_since = now
            return False
        if now - self._dead_air_since < 1.5:
            return False

        seg = self.pending
        if seg is None:
            self.status = "Finding the next record…"
            return True
        self._dead_air_since = 0.0
        log.warning("dead air recovered with %s", seg.track.display)
        with self._lock:
            seg.consumed = True
            self.pending = None
        self._prep_wake.set()
        self.engine.play(seg.track.path, seg.track.duration, fade_in=1.2)
        self._set_current(seg.track)
        self.phase = self.PHASE_PLAYING
        return True

    def _speech_trigger(self, seg: Segment) -> float:
        """Seconds before the end of the current record to open the mic."""
        if not seg.has_speech:
            return config.CROSSFADE_SECONDS
        return (
            config.CROSSFADE_SECONDS
            + config.SPEECH_OVERLAP_RATIO * seg.speech_seconds
            + config.SPEECH_TAIL_SECONDS
        )

    def _go_on_air(self) -> None:
        seg = self.pending
        if seg is None or not seg.ready.is_set():
            self.status = "Preparing the first record…"
            return
        with self._lock:
            seg.consumed = True
            self.pending = None
        self._prep_wake.set()

        # `current` is set before playback starts so preparation of the next
        # segment always sees the record it will be following.
        self._set_current(seg.track)
        self.engine.play(seg.track.path, seg.track.duration, fade_in=2.5)
        self.status = "On air"
        if seg.has_speech:
            self._speak(seg)
            self.phase = self.PHASE_HANDOVER
        else:
            self.phase = self.PHASE_PLAYING

    def _start_link(self, seg: Segment) -> None:
        self._speak(seg)
        # Bring the next record up once the DJ is most of the way through.
        lead = max(0.5, config.SPEECH_OVERLAP_RATIO * seg.speech_seconds)
        self._crossfade_at = time.monotonic() + lead
        self.phase = self.PHASE_TALKING
        log.debug("link on air, crossfade in %.1fs", lead)

    def _speak(self, seg: Segment) -> None:
        self.engine.speak(seg.items)
        if seg.script:
            self.last_script = seg.script
            with self._lock:
                self._speech_log.append((
                    dt.datetime.now().strftime("%H:%M"),
                    seg.script.text,
                ))
                del self._speech_log[:-40]

    def _handover(self, seg: Segment, silent: bool) -> None:
        with self._lock:
            seg.consumed = True
            self.pending = None
        self._prep_wake.set()
        self._set_current(seg.track)
        self.engine.crossfade_to(seg.track.path, seg.track.duration)
        self.phase = self.PHASE_HANDOVER
        log.info("now playing: %s%s", seg.track.display, "" if silent else "  (after link)")

    def _set_current(self, track: Track) -> None:
        with self._lock:
            self.current = track
            self.current_started = time.time()
            self._tracks_since_id += 1
        self.library.mark_played(track.path)

    # ------------------------------------------------------------------
    # engine events
    # ------------------------------------------------------------------
    def _on_engine_event(self, kind: str, payload: dict) -> None:
        if kind == "speech_started":
            with self._lock:
                self.on_air_text = payload.get("text", "")
                self.on_air_voice = payload.get("display", "")
        elif kind == "speech_finished":
            with self._lock:
                self.on_air_text = ""
                self.on_air_voice = ""

    # ------------------------------------------------------------------
    # controls / introspection
    # ------------------------------------------------------------------
    def skip(self) -> None:
        """Jump to the next record immediately, without a link."""
        seg = self.pending
        if seg is None or not seg.ready.is_set():
            log.info("skip ignored: nothing prepared yet")
            return
        self.engine.cancel_speech()
        self._handover(seg, silent=True)

    def set_volume(self, value: float) -> None:
        self.engine.set_volume(value)

    def snapshot(self) -> dict:
        state = self.engine.state()
        with self._lock:
            current = self.current
            pending = self.pending
            show = self.show
            speech_log = list(self._speech_log[-6:])
            on_air_text = self.on_air_text
            on_air_voice = self.on_air_voice
            status = self.status
            phase = self.phase
        scanned, total = self.library.scan_progress
        played = [t.display for t in self.director.history[:-1]][-3:][::-1]
        return {
            "recent": played,
            "profile": self.profile.key,
            "profile_label": self.profile.label,
            "model": self.brain.model,
            "running": self._running,
            "phase": phase,
            "status": status,
            "station": config.STATION_NAME,
            "frequency": config.STATION_FREQUENCY,
            "show": show.name if show else "",
            "tagline": show.tagline if show else "",
            "language": show.language if show else "",
            "language_label": show.language_label if show else "",
            "title": current.name if current else "",
            "artist": current.artist if current else "",
            "album": current.album if current else "",
            "year": current.year if current else "",
            "genre": current.genre if current else "",
            "cover": current.cover if current else "",
            "position": state.position,
            "duration": state.duration or (current.duration if current else 0.0),
            "remaining": state.remaining,
            "next_title": pending.track.display if pending else "",
            "next_ready": bool(pending and pending.ready.is_set()),
            "speaking": state.speaking,
            "on_air_text": on_air_text,
            "on_air_voice": on_air_voice,
            "crossfading": state.crossfading,
            "duck": state.duck,
            "vu": (state.vu_left, state.vu_right),
            "volume": state.volume,
            "brain": self.brain.detail if self.brain.available else "offline (fallback links)",
            "scanning": self.library.scanning,
            "scan_progress": (scanned, total),
            "speech_log": speech_log,
        }
