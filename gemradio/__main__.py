"""Command line entry point for GemRadio."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from . import config, lyrics
from .library import Library, preflight
from .logging_util import get_logger

log = get_logger(__name__)


def cmd_doctor() -> int:
    from .llm import Ollama

    print(f"\n  {config.STATION_NAME} — system check\n")
    ok = True

    def row(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"   [{'ok' if good else 'XX'}]  {label:<22} {detail}")

    row("ffmpeg", Path(config.FFMPEG).exists() or bool(config.FFMPEG), config.FFMPEG)
    row("music library", Path(config.MUSIC_ROOT).is_dir(), str(config.MUSIC_ROOT))

    lib = Library()
    stats = lib.stats()
    row("indexed tracks", stats["tracks"] > 0,
        f"{stats['tracks']} tracks · {stats['hours']:.0f} h · {stats['artists']} artists")

    voices = config.available_voices()
    row("piper voices", bool(voices), ", ".join(sorted(voices)) or "none found")
    for lang in ("en", "fr", "it"):
        vs = config.voices_for(lang)
        if vs:
            print(f"          {lang}: " + ", ".join(f"{v.display}({v.gender})" for v in vs))

    row("whisper", lyrics.available(),
        f"{Path(config.WHISPER_MODEL).name}" if lyrics.available() else "disabled/missing")

    brain = Ollama()
    row("gemma (ollama)", brain.check(), brain.detail)
    models = brain.installed()
    for m in sorted(models, key=lambda m: m["parameters"] or 0):
        print(f"          {m['name']:<22} {m['parameters'] or '?'}B"
              f"  {m['size'] / 1e9:.1f} GB")
    if models:
        print(f"          low power would use: {brain.smallest_model()}")

    try:
        from .audio import AudioEngine
        row("audio sink", True, " ".join(AudioEngine._choose_sink()[:1]))
    except Exception as exc:
        row("audio sink", False, str(exc))

    try:
        import PyQt5  # noqa: F401
        row("PyQt5", True, "")
    except ImportError:
        row("PyQt5", False, "pip install PyQt5")

    print()
    print("   Ready to broadcast.\n" if ok else "   Fix the items marked XX.\n")
    return 0 if ok else 1


def cmd_scan() -> int:
    lib = Library()
    print(f"Scanning {config.MUSIC_ROOT} …")
    last = [0.0]

    def progress(done: int, total: int) -> None:
        now = time.time()
        if now - last[0] > 0.5 or done == total:
            last[0] = now
            pct = 100.0 * done / total if total else 100.0
            print(f"\r  {done}/{total}  ({pct:5.1f}%)", end="", flush=True)

    lib.scan(progress=progress)
    print()
    stats = lib.stats()
    print(f"  {stats['tracks']} tracks · {stats['hours']:.0f} hours · "
          f"{stats['artists']} artists")
    return 0


def cmd_headless(minutes: float, profile: str | None = None) -> int:
    from .station import Station

    lib = Library()
    if lib.count() == 0:
        cmd_scan()
    station = Station(lib, profile=profile)

    def handler(*_):
        print("\nshutting down…")
        station.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    station.start()
    deadline = time.time() + minutes * 60 if minutes > 0 else float("inf")
    last_track = last_speech = ""
    try:
        while time.time() < deadline:
            s = station.snapshot()
            track = f"{s['show']} | {s['artist']} — {s['title']}"
            if track != last_track and s["title"]:
                last_track = track
                print(f"  ♪ {track}", flush=True)
            speech = s["on_air_text"]
            if speech and speech != last_speech:
                last_speech = speech
                print(f"  🎙 {s['on_air_voice']}: {speech}", flush=True)
            time.sleep(0.4)
    finally:
        station.shutdown()
    return 0


def cmd_gui(profile: str | None = None) -> int:
    problems = preflight()
    if problems:
        for p in problems:
            print(f"  !! {p}")
        print("\nRun 'python3 -m gemradio doctor' for details.")
        return 1
    from .gui import run
    return run(profile)


def main(argv: list[str] | None = None) -> int:
    # The profile flags are shared with every subcommand so they work both
    # before and after it: `gemradio --low-power headless` and
    # `gemradio headless --low-power` mean the same thing.
    # The defaults are suppressed rather than None: the subparser writes into
    # the same namespace as the main parser, so a real default here would wipe
    # out a flag that was given before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--profile", choices=sorted(config.PROFILES), default=argparse.SUPPRESS,
        help="full (default) or low: low uses the smallest installed model, "
             "talks less and skips the Whisper pass",
    )
    common.add_argument(
        "--low-power", action="store_const", const="low", dest="profile",
        default=argparse.SUPPRESS, help="shorthand for --profile low",
    )

    parser = argparse.ArgumentParser(
        prog="gemradio",
        description="GemRadio — a fully offline AI radio station.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("gui", parents=[common],
                   help="run the radio with its front panel (default)")
    sub.add_parser("doctor", parents=[common],
                   help="check that every component is present")
    sub.add_parser("scan", parents=[common],
                   help="index the music library and exit")
    head = sub.add_parser("headless", parents=[common],
                          help="broadcast in the terminal, no GUI")
    head.add_argument("--minutes", type=float, default=0,
                      help="stop after this many minutes (0 = forever)")

    args = parser.parse_args(argv)
    command = args.command or "gui"
    profile = getattr(args, "profile", None)
    if command == "doctor":
        return cmd_doctor()
    if command == "scan":
        return cmd_scan()
    if command == "headless":
        return cmd_headless(args.minutes, profile)
    return cmd_gui(profile)


if __name__ == "__main__":
    sys.exit(main())
