# Setting up GemRadio on a new machine

GemRadio runs entirely on your own computer. Once these steps are done it never
needs the internet again — but the setup itself downloads a language model, a
speech model and a few voices, so do this part online.

Everything below is written for Ubuntu/Debian. Other distributions work the
same way with their own package manager.

**Time:** about 20 minutes, most of it downloads.
**Disk:** roughly 12 GB (10 GB of that is the language model).

---

## 0. What the pieces are

| Piece | Does what | Required? |
|---|---|---|
| **ffmpeg** | decodes your music, resamples speech | yes |
| **PyQt5** | the front panel | yes (not for `headless`) |
| **Ollama + Gemma** | writes what the DJ says | no — falls back to fixed links |
| **Piper + voices** | speaks it | yes |
| **whisper.cpp** | listens to the next record | no — skipped if missing |

Only ffmpeg, PyQt5 and one Piper voice are truly required. The station starts
without the others and tells you what it is missing.

---

## 1. System packages

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-numpy python3-pyqt5 \
                    ffmpeg git build-essential cmake curl
```

Audio output goes through PipeWire or PulseAudio, which desktop Ubuntu already
has. GemRadio looks for `paplay`, then `pw-play`, then `aplay`. Check with:

```bash
pactl info      # should print a server name
```

On a headless server with no sound server, install `alsa-utils` and GemRadio
will use `aplay`.

---

## 2. Piper (the voices) — required

Piper is a small neural text-to-speech engine that runs fine on a CPU.

```bash
pip3 install --user piper-tts
```

Then fetch at least one voice. Voices are two files, `.onnx` and `.onnx.json`,
and they live in a folder tree that GemRadio reads directly:

```bash
VOICES="$HOME/piper_voices"
mkdir -p "$VOICES"/{en/en_GB,fr/fr_FR,it/it_IT}

BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main

# English female + English male (a two-host English show needs both)
mkdir -p "$VOICES/en/en_GB/jenny_dioco/medium" \
         "$VOICES/en/en_GB/northern_english_male/medium"
curl -L -o "$VOICES/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx" \
  "$BASE/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx"
curl -L -o "$VOICES/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json" \
  "$BASE/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json"
curl -L -o "$VOICES/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx" \
  "$BASE/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx"
curl -L -o "$VOICES/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx.json" \
  "$BASE/en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx.json"

# French — one model, two speakers (jessica = female, pierre = male)
mkdir -p "$VOICES/fr/fr_FR/upmc/medium"
curl -L -o "$VOICES/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx" \
  "$BASE/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx"
curl -L -o "$VOICES/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx.json" \
  "$BASE/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx.json"

# Italian
mkdir -p "$VOICES/it/it_IT/paola/medium"
curl -L -o "$VOICES/it/it_IT/paola/medium/it_IT-paola-medium.onnx" \
  "$BASE/it/it_IT/paola/medium/it_IT-paola-medium.onnx"
curl -L -o "$VOICES/it/it_IT/paola/medium/it_IT-paola-medium.onnx.json" \
  "$BASE/it/it_IT/paola/medium/it_IT-paola-medium.onnx.json"
```

Tell GemRadio where they are (skip if you used the default path already
configured in `config.py`):

```bash
export GEMRADIO_PIPER_VOICES="$HOME/piper_voices"
```

The directory layout matters — GemRadio reads the language from the path. Keep
the `<lang>/<locale>/<voice>/<quality>/` shape shown above.

Adding a voice later needs no code change: drop it in the right folder and it
appears. A language with both an `F` and an `M` voice can run two-host shows;
a language with one voice always runs solo.

---

## 3. Ollama + Gemma (the DJ's words) — optional but wanted

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma3:4b        # ~3 GB, comfortable on any machine
```

Bigger models write better links if you have the memory:

| Model | Parameters | Download | Needs |
|---|---|---|---|
| `gemma3:4b` | 4 B | ~3 GB | any machine, good for low power |
| `gemma3:12b` | 12 B | ~8 GB | 16 GB RAM |
| `gemma4:latest` | 8 B | ~10 GB | 16 GB RAM |
| `gemma4:26b` | 26 B | ~17 GB | 32 GB RAM |

Pick one and point the station at it:

```bash
export GEMRADIO_OLLAMA_MODEL=gemma3:4b
```

Low-power mode ignores this and always picks the **smallest model you have
installed**, so pulling a small one alongside a big one gives you a genuine
switch between them.

> **Important, and easy to miss:** Gemma 3 and 4 are *thinking* models. Asked a
> question normally they spend the whole token budget reasoning and hand back an
> empty answer. GemRadio always sends `"think": false`. If you write your own
> code against these models, do the same.

Skipping this step entirely is fine. Without Ollama the DJ still announces
records using built-in phrasing in all three languages; you just lose the
invented show names and the writing.

---

## 4. whisper.cpp (letting the DJ hear the next record) — optional

This is what lets the DJ react to what a song actually *is* rather than only
its tags.

### On a machine without a GPU

```bash
git clone https://github.com/ggerganov/whisper.cpp ~/whisper.cpp
cd ~/whisper.cpp
cmake -B build
cmake --build build -j --config Release
sh ./models/download-ggml-model.sh small     # multilingual, ~490 MB
```

### On a machine with an NVIDIA GPU

Install the CUDA toolkit first (`nvcc --version` should work), then:

```bash
git clone https://github.com/ggerganov/whisper.cpp ~/whisper.cpp
cd ~/whisper.cpp
cmake -B build -DGGML_CUDA=1
cmake --build build -j --config Release
sh ./models/download-ggml-model.sh small
```

Point GemRadio at the result if you put it somewhere else:

```bash
export GEMRADIO_WHISPER_BIN="$HOME/whisper.cpp/build/bin/whisper-cli"
export GEMRADIO_WHISPER_MODEL="$HOME/whisper.cpp/models/ggml-small.bin"
```

Use `ggml-base` instead of `small` on a slow machine — it is 5× faster and
still good enough to tell the DJ what a song is about. Or turn listening off:

```bash
export GEMRADIO_WHISPER=0
```

Listening never delays the music: it happens while the previous record plays,
and if it is not finished in time the station simply crossfades without a link.

---

## 5. GPU notes

### The context-length trap — read this if Ollama runs on the CPU

The single biggest performance mistake is invisible: **Ollama sizes its KV
cache from the context length, and a large context can push a model that would
otherwise fit in VRAM onto the CPU.** It does this silently. Nothing errors;
generation is simply four times slower.

Measured here on an RTX 4070 Laptop (8 GB) with `gemma4:12b-it-qat`:

| Context | Where it ran | VRAM | Speed |
|---|---|---|---|
| 16384 | **100% CPU** | — | 6.7 tok/s |
| 8192 | 76% GPU | 6.5 GB | **28 tok/s** |
| 4096 | 76% GPU | 6.4 GB | 29 tok/s |

The 16384 came from `OLLAMA_CONTEXT_LENGTH=16384` in the service unit — a
machine-wide default that silently applied to every model. GemRadio now sends
its own `num_ctx` on **every** request so it can never inherit a value that
costs it the GPU. Change it with `GEMRADIO_NUM_CTX` if you need a longer window.

Check what your own daemon defaults to:

```bash
systemctl show ollama --property=Environment | tr ' ' '\n' | grep -i context
```

On a laptop with switchable graphics there are two GPUs, an integrated one and
the discrete card. **Ollama only ever uses the NVIDIA card** — it needs CUDA and
cannot use an Intel or AMD integrated GPU — so if it reports a GPU at all, that
is the discrete one. `doctor` names it outright and shows the inference process
resident on it:

```
gpu: NVIDIA GeForce RTX 4070 Laptop GPU, 6495 MiB, 8188 MiB
gpu: resident: /usr/local/lib/ollama/llama-server, 6478 MiB
```

And confirm where a model actually landed — the `PROCESSOR` column is the truth:

```bash
ollama ps
# NAME               SIZE    PROCESSOR         CONTEXT
# gemma4:12b-it-qat  8.5 GB  24%/76% CPU/GPU   8192
```

`./run_gemradio.sh doctor` prints the same thing as `currently loaded on:`, and
the front panel shows it next to the model name in the status bar.

If a model still will not fit, use a smaller one, a smaller quantization, or
lower `GEMRADIO_NUM_CTX`. A partial offload like 76% is normal and still a
large win — the layers that fit run on the GPU.

### Everything else

**GemRadio does not need a GPU.** Every component works on a CPU.

- **Piper** synthesizes about 30× faster than real time on a CPU. GPU support
  exists but is not worth chasing; if ONNX Runtime cannot reach your GPU it
  prints a CUDA error and falls back to the CPU on its own. Force an attempt
  with `GEMRADIO_PIPER_CUDA=1`.
- **Ollama** uses the GPU automatically if the model fits in VRAM, and splits
  or falls back to CPU if it does not. An 8 GB card cannot hold a 12 B model,
  so it will run on the CPU — that is normal, and the station is built to hide
  the latency by preparing every link while the previous record is still
  playing.
- **whisper.cpp** needs the `-DGGML_CUDA=1` build above to use a GPU. A build
  made without it is CPU-only no matter what hardware you have, and it will not
  tell you. Check the binary rather than assuming:

  ```bash
  ldd $(which whisper-cli) | grep -ci cuda   # 0 means CPU-only
  ```

Check what actually happened:

```bash
ollama ps        # PROCESSOR column says CPU or GPU
nvidia-smi       # what is resident on the card
```

If your models run on the CPU and the machine feels loaded, that is precisely
what **low power** mode is for.

---

## 6. Your music

Point GemRadio at your library:

```bash
export GEMRADIO_MUSIC_DIR="$HOME/Music"
```

It reads mp3, flac, m4a, ogg, opus, wav, wma, aiff and more — anything ffmpeg
can decode. Cover art comes from tags embedded in the files, or from
`cover.jpg` / `folder.jpg` / `AlbumArt*.jpg` next to them.

Index it once (later runs refresh in the background automatically):

```bash
./run_gemradio.sh scan
```

Roughly 1,500 tracks a minute. The index lives in
`~/.local/share/gemradio/library.db`; deleting that file forces a full rescan.

---

## 7. Check and run

```bash
./run_gemradio.sh doctor    # every component, with what it found
./run_gemradio.sh           # the radio
```

`doctor` prints an `ok`/`XX` line per component, lists the voices per language
and every installed model, and tells you which one low power would choose.

Add it to your applications menu and dock:

```bash
./install_desktop.sh              # adds the icon
./install_desktop.sh --uninstall  # removes it again
```

Nothing is installed outside your home directory and nothing needs root.

---

## 8. Making the settings stick

Environment variables set in a terminal only last for that terminal. To keep
them, put them in `~/.bashrc` (or `~/.zshrc`):

```bash
cat >> ~/.bashrc <<'EOF'
export GEMRADIO_MUSIC_DIR="$HOME/Music"
export GEMRADIO_PIPER_VOICES="$HOME/piper_voices"
export GEMRADIO_OLLAMA_MODEL=gemma3:4b
EOF
```

For the **desktop icon**, which does not read your shell config, put them in
`Exec` instead — edit `~/.local/share/applications/gemradio.desktop`:

```ini
Exec=env GEMRADIO_MUSIC_DIR=/media/big-disk/music /path/to/run_gemradio.sh
```

---

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `no audio sink found` | No `paplay`/`pw-play`/`aplay`. `sudo apt install pulseaudio-utils alsa-utils` |
| OUTPUT dropdown is greyed out | Needs `paplay` or `pw-play`; with only `aplay` the station follows the system default |
| A Bluetooth speaker is missing from OUTPUT | Connect it first, then reopen the dropdown — the list is rebuilt each time |
| The DJ is too quiet under the music | Raise `GEMRADIO_VOICE_GAIN` (default 1.25) or lower `GEMRADIO_DUCK_LEVEL` (default 0.15) |
| Music plays, DJ never speaks | No Piper voice found. `./run_gemradio.sh doctor` lists what it sees; check `GEMRADIO_PIPER_VOICES` |
| DJ says only short generic lines | Ollama is not running. `ollama serve`, then check `curl localhost:11434/api/tags` |
| DJ says nothing at all and the log shows empty responses | You are calling a thinking model without `"think": false` |
| `CUDA failure 999` from Piper | Harmless — it falls back to the CPU. Leave `GEMRADIO_PIPER_CUDA` unset |
| No tracks found | Wrong `GEMRADIO_MUSIC_DIR`, or the files are shorter than `GEMRADIO_MIN_TRACK` (60 s) |
| Garbled non-Latin tags | Expected for old Windows-written files; GemRadio repairs cp1255/cp1251 automatically |
| Long gaps before the DJ speaks | CPU-bound models. Turn on low power, or `GEMRADIO_WHISPER=0` |
| Ollama keeps holding RAM after closing | Should not happen — the station unloads on exit. Check with `ollama ps`; `GEMRADIO_KEEP_ALIVE=0` makes it immediate |
| `ollama list` suddenly shows no models | A second `ollama serve` has taken port 11434 from the system service, and it looks in a different model directory. `pgrep -af "ollama serve"`, kill the stray one, then `systemctl start ollama` |
| Model runs on CPU despite a capable GPU | Context length — see [the GPU notes](#5-gpu-notes) |

Full log: `~/.local/share/gemradio/gemradio.log`. Run with `GEMRADIO_DEBUG=1`
for the verbose version.

---

## 10. Removing it

```bash
./install_desktop.sh --uninstall
rm -rf ~/.local/share/gemradio ~/.cache/gemradio
rm -rf /path/to/Gemma_Radio
```

Your music is never modified — GemRadio only ever reads it.
