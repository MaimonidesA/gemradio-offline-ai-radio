# GemRadio

A radio station that lives entirely on this machine. It plays your own music,
and between the records a DJ — Gemma 4 running in Ollama, speaking through
Piper voices — introduces what is coming, in English, French or Italian.

Nothing leaves the computer. No streaming, no lookups, no telemetry.

```bash
./run_gemradio.sh              # front panel
./run_gemradio.sh --low-power  # ...using the smallest model you have
./run_gemradio.sh doctor       # check every component
./run_gemradio.sh scan         # re-index the music library
./run_gemradio.sh headless     # broadcast in the terminal
./install_desktop.sh           # add it to the applications menu and dock
```

New here? **[SETUP.md](SETUP.md)** walks through installing this on any machine,
with and without a GPU.

## How it sounds

The point of the whole design is the handover. While a record plays, the next
one is already chosen, listened to by Whisper, written about by Gemma and
voiced by Piper. Then it is scheduled backwards from the end of the track:

```
                 speech starts                     crossfade begins
                      │                                   │
  ── current record ──┼───────────────────────────────────┼──────────▶
                      │  music ducks to −14 dB            │  next record rises
                      └── DJ line ────────────────────────┴───▶ (tail over intro)
```

The DJ opens the mic at `crossfade + 0.72 × speech_length + 4 s` before the end,
and the crossfade starts once the DJ is about 70% through the line — so the new
record comes up underneath the closing words and the old one is gone by the time
they finish. The ducking has real attack/hold/release, not a hard volume step.

## What runs where

| Part | What it uses |
|---|---|
| Music | `~/Music/מוזיקה`, indexed to SQLite with `ffprobe` |
| DJ script | `gemma4:latest` in Ollama, `think:false`, structured JSON output |
| Voices | Piper — EN Jenny/Alan, FR Jessica/Pierre, IT Paola |
| Listening | `whisper.cpp` + `ggml-small`, on a 20 s slice of the next track |
| Mixing | numpy mixer → `paplay`, dual decks, equal-power crossfade, ducking |
| Panel | PyQt5 |

**Gemma 4 must be called with `think:false`.** It is a thinking model: left on
its own it spends the entire token budget reasoning and returns an empty
message. Every call in `llm.py` disables thinking and asks for a JSON schema.

### Two hosts

English and French have both a female and a male voice, so those shows
sometimes run as a duet — two hosts trading a line each over the handover.
Italian has one voice, so Italian shows are always solo. The station never
tries to fake a voice it does not have.

## Low power

The **LOW POWER** button on the panel (or `--low-power`) changes how hard the
station works, while it stays on air:

| | Full | Low power |
|---|---|---|
| Model | your configured model | **smallest one installed** |
| DJ speaks | every 1–2 records | every 3–5 records |
| Voices at once | up to two | one |
| Whisper listening | on | off |
| Records per show | 5–9 | 10–16 |
| Show names | written by Gemma | from a curated list |

The point is that it plans a longer run of music at a time and thinks less
often, rather than degrading how it sounds. Switching mid-broadcast is safe:
anything already prepared still goes out, so you never hear the change.

Low power does not name a model — it asks Ollama for the **lightest one you
have** and uses that. Pull a small model alongside your big one and the button
becomes a real switch between them:

```bash
ollama pull gemma3:4b
```

## Shutting down cleanly

Closing the window stops the mixer and its ffmpeg decoders, ends the Piper
worker, halts any library scan, and **asks Ollama to unload the model** instead
of leaving several gigabytes resident for its keep-alive window. The same
happens on Ctrl-C, on `SIGTERM`, and on session logout — the handler is wired
to the window close, to Qt's `aboutToQuit`, and to an `atexit` hook, and is safe
to run more than once.

Check that nothing was left behind:

```bash
ollama ps                     # should list nothing
pgrep -af "piper_worker|paplay"   # should print nothing
```

### Shows

Records are grouped into blocks of 5–9 tracks. Each block is a "show" with a
name and a tagline invented by Gemma for the music in it and the time of day —
*Échos du Monde*, *Echi del Viaggio*, *The Long Wave*. The language is weighted
towards whatever Whisper actually heard in the upcoming records.

## Layout

```
run_gemradio.sh   launcher: starts Ollama if needed, clears proxy vars
install_desktop.sh  adds/removes the applications-menu entry and icon
SETUP.md          installing on a fresh machine, GPU and CPU
gemradio/
  assets/         the radio icon
  config.py       paths, voices, profiles, all tunables (env overridable)
  library.py      scanner, SQLite index, cover art, tag repair
  audio.py        the mixer: decks, crossfade, ducking, sink
  tts.py          Piper client, resampling, cache
  piper_worker.py Piper subprocess (isolated runtime)
  llm.py          Ollama client, JSON schemas
  lyrics.py       Whisper listening
  director.py     what plays next, who speaks, what they say
  station.py      the clock that ties it together
  gui.py          the front panel
```

## Tuning

Everything is an environment variable; nothing needs a code change.

```bash
GEMRADIO_PROFILE=low        ./run_gemradio.sh   # start in low power
GEMRADIO_KEEP_ALIVE=0       ./run_gemradio.sh   # never let Ollama hold the model
GEMRADIO_CROSSFADE=9        ./run_gemradio.sh   # longer crossfades
GEMRADIO_DUCK_LEVEL=0.12    ./run_gemradio.sh   # push music further down under speech
GEMRADIO_TALK_EVERY_MIN=2   ./run_gemradio.sh   # let the DJ talk less
GEMRADIO_WHISPER=0          ./run_gemradio.sh   # skip listening (saves CPU)
GEMRADIO_OLLAMA_MODEL=gemma4:12b-it-qat ./run_gemradio.sh
GEMRADIO_MUSIC_DIR=/some/other/library  ./run_gemradio.sh
GEMRADIO_DEBUG=1            ./run_gemradio.sh   # verbose log
```

Log: `~/.local/share/gemradio/gemradio.log`
Index: `~/.local/share/gemradio/library.db`
Caches: `~/.cache/gemradio/` (cover art, synthesized speech)

## If something is wrong

Run `./run_gemradio.sh doctor` first — it checks every dependency and prints
what it found.

- **No sound**: the station writes to `paplay`, falling back to `pw-play` and
  `aplay`. Check `pactl info`.
- **The DJ only says short generic lines**: Ollama is not reachable, so the
  station is using its built-in fallback links. It keeps broadcasting either way.
- **Piper prints a CUDA error**: harmless. This box falls back to CPU, which
  synthesizes about 30× faster than real time. Set `GEMRADIO_PIPER_CUDA=1` only
  if the GPU is working.
- **Long gaps before the DJ speaks**: Whisper and Gemma both run on CPU here.
  Preparation still finishes well inside a normal track; on very short tracks
  the station simply crossfades without a link rather than waiting.
