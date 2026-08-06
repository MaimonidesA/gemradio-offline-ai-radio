"""Programme direction: what plays next, who speaks, and what they say.

The director owns the editorial side of the station.  It keeps the music
rotation varied, groups tracks into shows with their own name and language, and
turns track metadata (plus anything Whisper heard) into a short DJ script.

Every LLM-backed decision has a deterministic fallback, so the station keeps
broadcasting even if Ollama is stopped mid-show.
"""

from __future__ import annotations

import datetime as dt
import random
import re
import threading
from dataclasses import dataclass, field

from . import config
from .library import Library, Track
from .llm import SCRIPT_SCHEMA, SHOW_SCHEMA, Ollama
from .logging_util import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Show identity
# --------------------------------------------------------------------------

@dataclass
class Show:
    name: str
    tagline: str
    language: str
    tracks_left: int = 6

    @property
    def language_label(self) -> str:
        return config.LANGUAGE_NAMES.get(self.language, self.language.upper())


FALLBACK_SHOWS = {
    "en": [("The Long Wave", "music for the small hours"),
           ("Amber Hour", "warm records, low light"),
           ("Night Signal", "everything that drifts"),
           ("The Listening Room", "one record after another")],
    "fr": [("Nocturne Bleu", "la musique jusqu'au matin"),
           ("Le Grand Bain", "des disques qui respirent"),
           ("Heure Douce", "pour la fin de la journée"),
           ("La Dérive", "sans frontières")],
    "it": [("Notte Lunga", "musica fino all'alba"),
           ("Onde Calme", "dischi da ascoltare piano"),
           ("Ora Blu", "il suono della sera"),
           ("Il Salotto", "un disco dopo l'altro")],
}


def part_of_day(now: dt.datetime | None = None) -> str:
    h = (now or dt.datetime.now()).hour
    if h < 5:
        return "the middle of the night"
    if h < 9:
        return "early morning"
    if h < 12:
        return "late morning"
    if h < 15:
        return "the afternoon"
    if h < 19:
        return "early evening"
    if h < 23:
        return "the evening"
    return "late night"


# --------------------------------------------------------------------------
# Scripts
# --------------------------------------------------------------------------

@dataclass
class Script:
    lines: list[tuple[str, str]] = field(default_factory=list)   # (voice_key, text)
    kind: str = "transition"
    source: str = "fallback"

    @property
    def text(self) -> str:
        return " ".join(t for _, t in self.lines)


STATION_STYLE = """You are the voice of {station}, an independent radio station \
that plays one record after another from a personal music collection: jazz, \
bossa nova, chanson, blues, folk, world music, classical and more.

House style — this matters:
- Warm, unhurried, curious. Never shouty, never salesy, no radio cliches.
- Speak like a person who loves the record, not like an announcer reading ads.
- SHORT. Each line is one or two sentences, {words} words at most.
- Talk about mood, texture, the time of day, the move from one track to the next.
- Use ONLY the facts you are given. Never invent biography, chart positions, \
recording dates, anecdotes or awards. If you know nothing about a record, \
speak about how it sounds instead.
- No emoji, no stage directions, no asterisks, no quotation marks around your \
own speech. Plain spoken words only, since this is read aloud.
- Never mention that you are an AI or a language model.
- Write in {language_name}. Every word must be in {language_name}."""

LANGUAGE_FULL = {"en": "English", "fr": "French", "it": "Italian"}


class Director:
    def __init__(self, library: Library, brain: Ollama | None = None,
                 profile: config.Profile | None = None):
        self.library = library
        self.brain = brain or Ollama()
        self.profile = profile or config.PROFILES[config.DEFAULT_PROFILE]
        self.rng = random.Random()
        self._lock = threading.Lock()

        self.recent_paths: list[str] = []
        self.recent_artists: list[str] = []
        self.history: list[Track] = []
        self.show: Show | None = None
        self.shows_played = 0
        self._pool: list[Track] = []
        self._talk_countdown = 1

    # -- rotation -----------------------------------------------------------
    def _refill_pool(self) -> None:
        pool = self.library.playable(limit=1500)
        self.rng.shuffle(pool)
        self._pool = pool
        log.debug("director: pool refilled with %d tracks", len(pool))

    def next_track(self) -> Track | None:
        with self._lock:
            for _ in range(3):
                if len(self._pool) < 30:
                    self._refill_pool()
                if not self._pool:
                    return None
                choice = self._pick_from_pool()
                if choice is not None:
                    self._remember(choice)
                    return choice
                self._refill_pool()
            return None

    def _pick_from_pool(self) -> Track | None:
        best: Track | None = None
        for idx, track in enumerate(self._pool[:400]):
            if track.path in self.recent_paths:
                continue
            artist = (track.artist or track.folder).lower()
            if artist and artist in self.recent_artists:
                continue
            best = track
            self._pool.pop(idx)
            break
        if best is None and self._pool:
            best = self._pool.pop(0)
        return best

    def _remember(self, track: Track) -> None:
        self.recent_paths.append(track.path)
        if len(self.recent_paths) > config.TRACK_HISTORY:
            self.recent_paths.pop(0)
        artist = (track.artist or track.folder).lower()
        if artist:
            self.recent_artists.append(artist)
            if len(self.recent_artists) > config.ARTIST_COOLDOWN:
                self.recent_artists.pop(0)
        self.history.append(track)
        if len(self.history) > 40:
            self.history.pop(0)

    def peek_upcoming(self, n: int = 4) -> list[Track]:
        with self._lock:
            if len(self._pool) < n:
                self._refill_pool()
            return self._pool[:n]

    # -- shows --------------------------------------------------------------
    def _language_hint(self, upcoming: list[Track]) -> list[str]:
        """Languages Whisper actually heard in the records coming up."""
        heard = [t.lyric_lang for t in upcoming if t.lyric_lang in ("en", "fr", "it")]
        available = set(config.available_languages())
        return [l for l in dict.fromkeys(heard) if l in available]

    def start_new_show(self, upcoming: list[Track]) -> Show:
        languages = config.available_languages()
        length = self.rng.randint(*self.profile.block_tracks)

        # Keep a language from repeating twice in a row when we have a choice,
        # and lean towards whatever the upcoming records are actually sung in.
        prev = self.show.language if self.show else ""
        options = [l for l in languages if l != prev] or languages
        heard = self._language_hint(upcoming)
        weights = [
            (3.0 if l == "en" else 2.0) * (2.5 if l in heard else 1.0)
            for l in options
        ]
        language = self.rng.choices(options, weights=weights, k=1)[0]

        show = self._llm_show(upcoming, language, length)
        if show is None:
            name, tagline = self.rng.choice(FALLBACK_SHOWS.get(language, FALLBACK_SHOWS["en"]))
            show = Show(name=name, tagline=tagline, language=language, tracks_left=length)
        self.show = show
        self.shows_played += 1
        log.info("show: %s (%s) — %s", show.name, show.language, show.tagline)
        return show

    def _llm_show(self, upcoming: list[Track], language: str, length: int) -> Show | None:
        # In low power the show identity comes from the curated list instead of
        # costing a model call of its own.
        if not self.brain.available or not self.profile.llm_show_names:
            return None
        listing = "\n".join(f"- {t.describe()}" for t in upcoming[:6]) or "- a varied selection"
        system = (
            f"You name radio programmes for {config.STATION_NAME}. "
            f"Answer in {LANGUAGE_FULL[language]}. The show name is 1-3 words, evocative, "
            "never generic like 'Music Hour'. The tagline is at most 6 words, lowercase. "
            "Do not invent facts about the artists."
        )
        user = (
            f"It is {part_of_day()}. Name the programme that will play these records:\n"
            f"{listing}\n\nRespond with show_name, tagline, and language set to '{language}'."
        )
        data = self.brain.chat_json(system, user, SHOW_SCHEMA, temperature=1.0, num_predict=160)
        if not data:
            return None
        name = _clean_speech(str(data.get("show_name", "")))[:40]
        tagline = _clean_speech(str(data.get("tagline", "")))[:60]
        if not name:
            return None
        return Show(name=name, tagline=tagline, language=language, tracks_left=length)

    def tick_show(self, upcoming: list[Track]) -> Show:
        """Advance the block counter, starting a new show when one runs out."""
        if self.show is None or self.show.tracks_left <= 0:
            return self.start_new_show(upcoming)
        self.show.tracks_left -= 1
        return self.show

    # -- talk cadence -------------------------------------------------------
    def should_talk(self) -> bool:
        self._talk_countdown -= 1
        if self._talk_countdown <= 0:
            low, high = self.profile.talk_every
            self._talk_countdown = self.rng.randint(low, high)
            return True
        return False

    # -- host casting -------------------------------------------------------
    def hosts(self, language: str) -> tuple[str | None, str | None]:
        female = config.pick_voice(language, "F")
        male = config.pick_voice(language, "M")
        f_key = female.key if female and female.gender == "F" else None
        m_key = male.key if male and male.gender == "M" else None
        return f_key, m_key

    def can_duet(self, language: str) -> bool:
        f, m = self.hosts(language)
        return bool(f and m)

    # -- scripts ------------------------------------------------------------
    def build_script(self, kind: str, current: Track | None, upcoming: Track | None,
                     show: Show) -> Script:
        language = show.language
        f_key, m_key = self.hosts(language)
        solo = f_key or m_key
        if solo is None:
            return Script(lines=[], kind=kind, source="no-voice")

        duet = (
            self.profile.allow_duet
            and kind in {"transition", "open"}
            and self.can_duet(language)
            and self.rng.random() < (0.45 if language == "en" else 0.25)
        )

        lines = self._llm_script(kind, current, upcoming, show, duet)
        source = "gemma"
        if not lines:
            lines = self._fallback_lines(kind, current, upcoming, show)
            source = "fallback"

        voiced: list[tuple[str, str]] = []
        first_speaker = self.rng.choice(["F", "M"]) if duet else ("F" if f_key else "M")
        for idx, (speaker, text) in enumerate(lines):
            if duet:
                want = speaker if speaker in ("F", "M") else ("F" if idx % 2 == 0 else "M")
                if idx == 0:
                    want = first_speaker
                key = (f_key if want == "F" else m_key) or solo
            else:
                key = solo
            voiced.append((key, text))
        return Script(lines=voiced, kind=kind, source=source)

    def _llm_script(self, kind: str, current: Track | None, upcoming: Track | None,
                    show: Show, duet: bool) -> list[tuple[str, str]]:
        if not self.brain.available:
            return []
        language = show.language
        words = 30 if duet else 40
        system = STATION_STYLE.format(
            station=config.STATION_NAME,
            words=words,
            language_name=LANGUAGE_FULL[language],
        )
        if duet:
            system += (
                "\n\nTwo hosts are on air: F (a woman) and M (a man). Write exactly two "
                "lines that answer each other naturally, alternating speakers. They are "
                "colleagues who like each other; keep it light and brief. The FIRST line "
                "reacts to the record that is ending; the SECOND line must announce the "
                "record that is coming next and say its artist and title."
            )
        else:
            system += (
                "\n\nOne host is on air. Write exactly one line, and it must name the "
                "record coming next."
            )

        system += (
            f"\n\nThe programme is called '{show.name}'"
            + (f" — {show.tagline}." if show.tagline else ".")
        )

        parts = [f"Time of day: {part_of_day()}."]
        if kind == "open":
            parts.append(
                f"This is the top of the programme. Welcome listeners to "
                f"{config.STATION_NAME} on {config.STATION_FREQUENCY} and to '{show.name}', "
                "then lead into the first record."
            )
        elif kind == "station_id":
            parts.append(
                f"Give a brief station identification for {config.STATION_NAME} "
                f"{config.STATION_FREQUENCY}, mention the programme name, then hand over "
                "to the music."
            )
        else:
            parts.append(
                "This is the handover between two records. Close the one that is ending "
                "and bring in the next one."
            )

        if current is not None:
            parts.append(f"Record now ending: {current.describe()}")
        if upcoming is not None:
            parts.append(f"Record coming next: {upcoming.describe()}")
            if upcoming.lyric:
                parts.append(
                    "A snippet actually heard in the coming record: "
                    f'"{upcoming.lyric[:240]}". Use it only if it inspires a natural '
                    "remark; never read it out as a quotation."
                )
        if self.history[:-1]:
            recent = ", ".join(t.display for t in self.history[-4:-1])
            if recent:
                parts.append(f"Earlier in this programme: {recent}.")

        if upcoming is not None:
            parts.append(
                f'Name the next record inside your speech, naturally: the artist is '
                f'{upcoming.artist or "unknown"} and the title is "{upcoming.name}". '
                "Use that exact title — never put the album name in its place, and "
                "never change or translate the title."
            )
        else:
            parts.append("Do not invent a title.")

        data = self.brain.chat_json(
            system, "\n".join(parts), SCRIPT_SCHEMA,
            temperature=0.98, num_predict=self.profile.num_predict,
        )
        if not data:
            return []
        raw = data.get("lines") or []
        out: list[tuple[str, str]] = []
        for item in raw[: (2 if duet else 1)]:
            if not isinstance(item, dict):
                continue
            text = _clean_speech(str(item.get("text", "")))
            if not text:
                continue
            speaker = str(item.get("speaker", "F")).upper()[:1]
            out.append((speaker if speaker in ("F", "M") else "F", _limit_words(text, 55)))
        return out

    def _fallback_lines(self, kind: str, current: Track | None,
                        upcoming: Track | None, show: Show) -> list[tuple[str, str]]:
        lang = show.language
        nxt = upcoming.display if upcoming else ""
        cur = current.display if current else ""
        station = f"{config.STATION_NAME} {config.STATION_FREQUENCY}"

        if lang == "fr":
            if kind == "open":
                text = f"Vous écoutez {station}. Voici {show.name}."
            elif kind == "station_id":
                text = f"Ici {station}, toujours {show.name}."
            elif nxt:
                text = f"C'était {cur}. Et maintenant, {nxt}." if cur else f"Voici {nxt}."
            else:
                text = f"On continue sur {station}."
        elif lang == "it":
            if kind == "open":
                text = f"State ascoltando {station}. Comincia {show.name}."
            elif kind == "station_id":
                text = f"Qui {station}, siamo dentro {show.name}."
            elif nxt:
                text = f"Era {cur}. E adesso, {nxt}." if cur else f"Ecco {nxt}."
            else:
                text = f"Si continua su {station}."
        else:
            if kind == "open":
                text = f"You're listening to {station}. This is {show.name}."
            elif kind == "station_id":
                text = f"This is {station}, still inside {show.name}."
            elif nxt:
                text = f"That was {cur}. And now, {nxt}." if cur else f"Here's {nxt}."
            else:
                text = f"Staying with us here on {station}."
        return [("F", text)]


# --------------------------------------------------------------------------
# Text hygiene for speech
# --------------------------------------------------------------------------

_STAGE = re.compile(r"[*_`#]|\[[^\]]*\]|\([^)]*\)")
_SPEAKER_TAG = re.compile(r"^\s*(F|M|host\s*[12fm]?|speaker\s*\d?)\s*[:\-–]\s*", re.I)


def _clean_speech(text: str) -> str:
    text = _STAGE.sub(" ", text or "")
    text = _SPEAKER_TAG.sub("", text)
    text = text.replace('"', "").replace("“", "").replace("”", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    clipped = " ".join(words[:max_words])
    for stop in (". ", "! ", "? "):
        idx = clipped.rfind(stop)
        if idx > len(clipped) * 0.4:
            return clipped[: idx + 1]
    return clipped.rstrip(",;:") + "."
