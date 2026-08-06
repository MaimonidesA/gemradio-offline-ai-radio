"""Music library indexing.

Walks the music root, reads tags with ffprobe, resolves cover art and stores
everything in SQLite.  The scan is incremental (mtime based) and runs in a
background thread so the radio can go on air before indexing finishes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from . import config
from .logging_util import get_logger

log = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    path        TEXT PRIMARY KEY,
    folder      TEXT,
    title       TEXT,
    artist      TEXT,
    album       TEXT,
    genre       TEXT,
    year        TEXT,
    duration    REAL,
    mtime       REAL,
    size        INTEGER,
    cover       TEXT,
    lyric       TEXT,
    lyric_lang  TEXT,
    play_count  INTEGER DEFAULT 0,
    last_played REAL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist);
CREATE INDEX IF NOT EXISTS idx_tracks_folder ON tracks(folder);
CREATE INDEX IF NOT EXISTS idx_tracks_played ON tracks(last_played);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


# --------------------------------------------------------------------------
# Tag text repair
# --------------------------------------------------------------------------

_HEBREW = re.compile(r"[֐-׿]")
# Tags written by old Windows players are frequently cp1255/cp1251 bytes that
# ffprobe hands back decoded as latin-1.  Round-tripping recovers the original.
_MOJIBAKE_HINT = re.compile(r"[À-ÿ]{2,}")


def repair_text(value: str) -> str:
    if not value:
        return ""
    text = value.strip()
    if not text or not _MOJIBAKE_HINT.search(text):
        return text
    for codec in ("cp1255", "cp1251"):
        try:
            candidate = text.encode("latin-1").decode(codec)
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if codec == "cp1255" and _HEBREW.search(candidate):
            return candidate.strip()
        if codec == "cp1251" and re.search(r"[Ѐ-ӿ]", candidate):
            return candidate.strip()
    return text


def _clean(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", repair_text(str(value))).strip()


def _title_from_filename(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"^\s*\d{1,3}\s*[-._) ]+", "", stem)   # leading track number
    stem = re.sub(r"[_]+", " ", stem)
    return re.sub(r"\s+", " ", stem).strip() or path.stem


# Release-scene noise that ends up in folder names and would otherwise be read
# out on air: "Queen discography (MP3@320Kbps)" must become "Queen".
_RELEASE_JUNK = re.compile(
    r"""(?ix)
    \s*[\[(][^\])]*(?:mp3|flac|kbps|vbr|cbr|\d{3}\s*k|rip|www\.|encoded|
                        discography|complete|remaster(?:ed)?|box\s*set)[^\])]*[\])]
  | \s*\b(?:mp3|flac|ape|wav)\s*@?\s*\d{2,3}\s*kbps\b
  | \s*\bdiscography\b
  | \s*[\[(]\s*\d{4}\s*(?:-\s*\d{4})?\s*[\])]
  | \s*\bwww\.[^\s]+
    """
)


def clean_artist(name: str) -> str:
    """Make a folder-derived artist name safe to say out loud."""
    if not name:
        return ""
    cleaned = _RELEASE_JUNK.sub(" ", name)
    cleaned = re.sub(r"\s*[-–_]\s*$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–_,")
    return cleaned or name.strip()


def _artist_from_path(path: Path, root: Path) -> str:
    """Guess the artist from the first directory below the music root."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return ""
    parts = rel.parts[:-1]
    if not parts:
        return ""
    return clean_artist(repair_text(parts[0]))


# --------------------------------------------------------------------------
# Track record
# --------------------------------------------------------------------------

@dataclass
class Track:
    path: str
    folder: str = ""
    title: str = ""
    artist: str = ""
    album: str = ""
    genre: str = ""
    year: str = ""
    duration: float = 0.0
    cover: str = ""
    lyric: str = ""
    lyric_lang: str = ""
    play_count: int = 0
    last_played: float = 0.0

    @property
    def name(self) -> str:
        return self.title or Path(self.path).stem

    @property
    def display(self) -> str:
        if self.artist:
            return f"{self.artist} — {self.name}"
        return self.name

    def describe(self) -> str:
        """Compact one-line description handed to the DJ model."""
        bits = [f'"{self.name}"']
        if self.artist:
            bits.append(f"by {self.artist}")
        if self.album:
            bits.append(f"from the album '{self.album}'")
        if self.year:
            bits.append(f"({self.year})")
        if self.genre:
            bits.append(f"[{self.genre}]")
        if self.duration:
            bits.append(f"{int(self.duration // 60)}:{int(self.duration % 60):02d}")
        return " ".join(bits)


def _row_to_track(row: sqlite3.Row) -> Track:
    # Cleaning on read as well as on write means an index built by an earlier
    # version does not need a full rescan to stop saying "(MP3@320Kbps)".
    return Track(
        path=row["path"], folder=row["folder"] or "", title=row["title"] or "",
        artist=clean_artist(row["artist"] or ""), album=row["album"] or "",
        genre=row["genre"] or "",
        year=row["year"] or "", duration=row["duration"] or 0.0, cover=row["cover"] or "",
        lyric=row["lyric"] or "", lyric_lang=row["lyric_lang"] or "",
        play_count=row["play_count"] or 0, last_played=row["last_played"] or 0.0,
    )


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------

def probe_file(path: Path, root: Path) -> dict | None:
    cmd = [
        config.FFPROBE, "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=30).stdout
        data = json.loads(out or b"{}")
    except Exception:
        return None

    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        return None

    try:
        duration = float(fmt.get("duration") or audio.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    tags = {str(k).lower(): v for k, v in (fmt.get("tags") or {}).items()}
    for s in streams:
        for k, v in (s.get("tags") or {}).items():
            tags.setdefault(str(k).lower(), v)

    has_pic = any(
        s.get("codec_type") == "video"
        and (s.get("disposition") or {}).get("attached_pic")
        for s in streams
    )

    year = _clean(tags.get("date") or tags.get("year") or tags.get("originalyear"))
    m = re.search(r"(1[89]\d{2}|20\d{2})", year)
    year = m.group(1) if m else ""

    return {
        "duration": duration,
        "title": _clean(tags.get("title")) or _title_from_filename(path),
        "artist": _clean(tags.get("artist") or tags.get("album_artist") or tags.get("performer"))
        or _artist_from_path(path, root),
        "album": _clean(tags.get("album")),
        "genre": _clean(tags.get("genre")),
        "year": year,
        "has_pic": has_pic,
    }


# --------------------------------------------------------------------------
# Cover art
# --------------------------------------------------------------------------

def _folder_cover(folder: Path) -> str:
    try:
        images = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    except OSError:
        return ""
    if not images:
        return ""

    def score(p: Path) -> tuple:
        low = p.name.lower()
        hint = next((i for i, h in enumerate(config.IMAGE_NAME_HINTS) if h in low), 99)
        small = 1 if "small" in low else 0
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        return (hint, small, -size)

    images.sort(key=score)
    return str(images[0])


def extract_cover(path: Path, has_pic: bool) -> str:
    """Return a filesystem path to artwork for this track ('' if none)."""
    folder_art = _folder_cover(path.parent)
    if folder_art:
        return folder_art
    if not has_pic:
        return ""

    digest = hashlib.sha1(str(path).encode("utf-8", "replace")).hexdigest()[:20]
    out = config.COVER_DIR / f"{digest}.jpg"
    if out.is_file():
        return str(out)
    cmd = [
        config.FFMPEG, "-v", "quiet", "-y", "-i", str(path),
        "-an", "-vframes", "1", "-vf", "scale=480:-1", str(out),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
    except Exception:
        return ""
    return str(out) if out.is_file() and out.stat().st_size > 0 else ""


# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------

class Library:
    def __init__(self, root: Path | None = None, db_path: Path | None = None):
        config.ensure_dirs()
        self.root = Path(root or config.MUSIC_ROOT)
        self.db_path = Path(db_path or config.DB_PATH)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self.scan_progress: tuple[int, int] = (0, 0)
        self.scanning = False
        self._stop = threading.Event()
        with self._conn() as c:
            c.executescript(SCHEMA)

    # -- connection handling ------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    # -- scanning -----------------------------------------------------------
    def iter_audio_files(self) -> Iterable[Path]:
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                if Path(fn).suffix.lower() in config.AUDIO_EXTENSIONS:
                    yield Path(dirpath) / fn

    def count(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM tracks").fetchone()[0])

    def scan(self, progress: Callable[[int, int], None] | None = None,
             workers: int = 8) -> int:
        """Incremental scan.  Returns the number of tracks added or refreshed."""
        self.scanning = True
        self._stop.clear()
        try:
            known: dict[str, float] = {
                r["path"]: (r["mtime"] or 0.0)
                for r in self._conn().execute("SELECT path, mtime FROM tracks")
            }
            files = list(self.iter_audio_files())
            seen: set[str] = set()
            todo: list[Path] = []
            for p in files:
                sp = str(p)
                seen.add(sp)
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                if abs(known.get(sp, -1.0) - mtime) > 1.0:
                    todo.append(p)

            stale = [p for p in known if p not in seen]
            if stale:
                with self._write_lock:
                    conn = self._conn()
                    conn.executemany("DELETE FROM tracks WHERE path = ?",
                                     [(p,) for p in stale])
                    conn.commit()
                log.info("library: dropped %d missing tracks", len(stale))

            total = len(todo)
            self.scan_progress = (0, total)
            if progress:
                progress(0, total)
            if not total:
                log.info("library: up to date (%d tracks)", len(seen))
                return 0

            log.info("library: indexing %d new/changed files", total)
            done = 0
            batch: list[tuple] = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for path, info in zip(todo, pool.map(self._probe_one, todo)):
                    done += 1
                    self.scan_progress = (done, total)
                    if info is not None:
                        batch.append(info)
                    if len(batch) >= 100:
                        self._store(batch)
                        batch.clear()
                    if progress and done % 25 == 0:
                        progress(done, total)
                    if self._stop.is_set():
                        break
            if batch:
                self._store(batch)
            if progress:
                progress(done, total)
            log.info("library: indexed %d files (%d tracks total)", done, self.count())
            return done
        finally:
            self.scanning = False
            self.scan_progress = (self.scan_progress[1], self.scan_progress[1])

    def _probe_one(self, path: Path) -> tuple | None:
        if self._stop.is_set():
            return None
        info = probe_file(path, self.root)
        if not info or info["duration"] <= 0:
            return None
        try:
            st = path.stat()
        except OSError:
            return None
        cover = extract_cover(path, info["has_pic"])
        return (
            str(path), str(path.parent), info["title"], info["artist"], info["album"],
            info["genre"], info["year"], info["duration"], st.st_mtime, st.st_size, cover,
        )

    def _store(self, rows: list[tuple]) -> None:
        with self._write_lock:
            conn = self._conn()
            conn.executemany(
                """INSERT INTO tracks
                   (path, folder, title, artist, album, genre, year, duration,
                    mtime, size, cover)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(path) DO UPDATE SET
                     folder=excluded.folder, title=excluded.title,
                     artist=excluded.artist, album=excluded.album,
                     genre=excluded.genre, year=excluded.year,
                     duration=excluded.duration, mtime=excluded.mtime,
                     size=excluded.size, cover=excluded.cover""",
                rows,
            )
            conn.commit()

    def scan_async(self, progress: Callable[[int, int], None] | None = None) -> threading.Thread:
        t = threading.Thread(target=self._scan_safe, args=(progress,),
                             name="library-scan", daemon=True)
        t.start()
        return t

    def _scan_safe(self, progress) -> None:
        try:
            self.scan(progress)
        except Exception:
            log.exception("library scan failed")

    def stop_scan(self) -> None:
        self._stop.set()

    # -- queries ------------------------------------------------------------
    def playable(self, limit: int = 4000) -> list[Track]:
        rows = self._conn().execute(
            """SELECT * FROM tracks
               WHERE duration BETWEEN ? AND ?
               ORDER BY last_played ASC, RANDOM()
               LIMIT ?""",
            (config.MIN_TRACK_SECONDS, config.MAX_TRACK_SECONDS, limit),
        ).fetchall()
        return [_row_to_track(r) for r in rows]

    def random_tracks(self, n: int = 40) -> list[Track]:
        rows = self._conn().execute(
            """SELECT * FROM tracks
               WHERE duration BETWEEN ? AND ?
               ORDER BY RANDOM() LIMIT ?""",
            (config.MIN_TRACK_SECONDS, config.MAX_TRACK_SECONDS, n),
        ).fetchall()
        return [_row_to_track(r) for r in rows]

    def get(self, path: str) -> Track | None:
        row = self._conn().execute("SELECT * FROM tracks WHERE path = ?", (path,)).fetchone()
        return _row_to_track(row) if row else None

    def mark_played(self, path: str) -> None:
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                "UPDATE tracks SET play_count = play_count + 1, last_played = ? WHERE path = ?",
                (time.time(), path),
            )
            conn.commit()

    def store_lyric(self, path: str, lyric: str, lang: str) -> None:
        with self._write_lock:
            conn = self._conn()
            conn.execute("UPDATE tracks SET lyric = ?, lyric_lang = ? WHERE path = ?",
                         (lyric, lang, path))
            conn.commit()

    def stats(self) -> dict:
        conn = self._conn()
        row = conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(duration),0) d, COUNT(DISTINCT artist) a FROM tracks"
        ).fetchone()
        return {"tracks": row["n"], "hours": (row["d"] or 0) / 3600.0, "artists": row["a"]}


def preflight() -> list[str]:
    """Human readable problems that would stop the station from working."""
    problems = []
    if not shutil.which(config.FFMPEG):
        problems.append("ffmpeg not found")
    if not shutil.which(config.FFPROBE):
        problems.append("ffprobe not found")
    if not Path(config.MUSIC_ROOT).is_dir():
        problems.append(f"music directory missing: {config.MUSIC_ROOT}")
    if not config.available_voices():
        problems.append(f"no Piper voices under {config.PIPER_VOICES_DIR}")
    return problems
