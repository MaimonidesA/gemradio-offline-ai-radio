#!/usr/bin/env python3
"""Persistent Piper synthesis worker.

Runs as its own process with PYTHONPATH pointed at the Piper runtime, so the
station never has to care about that runtime's numpy/onnxruntime versions.
Voices stay loaded between requests.

Protocol: one JSON object per line on stdin, one JSON object per line on
stdout.  Only stdout carries protocol data; everything else goes to stderr.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import wave

_VOICES: dict[tuple[str, bool], object] = {}


def _log(msg: str) -> None:
    print(f"[piper_worker] {msg}", file=sys.stderr, flush=True)


def _load(model: str, use_cuda: bool):
    key = (model, use_cuda)
    voice = _VOICES.get(key)
    if voice is not None:
        return voice
    from piper import PiperVoice

    voice = PiperVoice.load(model, use_cuda=use_cuda)
    _VOICES[key] = voice
    _log(f"loaded {os.path.basename(model)} (cuda={use_cuda})")
    return voice


def _synthesize(req: dict) -> dict:
    from piper import SynthesisConfig

    model = req["model"]
    text = (req.get("text") or "").strip()
    out_path = req["out"]
    if not text:
        return {"ok": False, "error": "empty text"}

    voice = _load(model, bool(req.get("use_cuda")))
    syn = SynthesisConfig(
        speaker_id=req.get("speaker_id"),
        length_scale=req.get("length_scale"),
        noise_scale=req.get("noise_scale"),
        noise_w_scale=req.get("noise_w_scale"),
        normalize_audio=True,
    )

    sentence_silence = float(req.get("sentence_silence", 0.22))
    frames = 0
    sample_rate = 22050
    with wave.open(out_path, "wb") as wav:
        silence = b""
        first = True
        for chunk in voice.synthesize(text, syn_config=syn):
            if first:
                sample_rate = chunk.sample_rate
                wav.setframerate(chunk.sample_rate)
                wav.setsampwidth(chunk.sample_width)
                wav.setnchannels(chunk.sample_channels)
                silence = bytes(
                    int(chunk.sample_rate * sentence_silence)
                    * chunk.sample_width * chunk.sample_channels
                )
                first = False
            elif silence:
                wav.writeframes(silence)
                frames += len(silence) // (chunk.sample_width * chunk.sample_channels)
            audio = chunk.audio_int16_bytes
            wav.writeframes(audio)
            frames += len(audio) // (chunk.sample_width * chunk.sample_channels)
        if first:
            return {"ok": False, "error": "piper produced no audio"}

    return {"ok": True, "duration": frames / float(sample_rate),
            "sample_rate": sample_rate, "out": out_path}


def main() -> int:
    _log("ready")
    print(json.dumps({"ok": True, "event": "ready"}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = req.get("id")
        if req.get("cmd") == "quit":
            break
        try:
            resp = _synthesize(req)
        except Exception as exc:
            _log(traceback.format_exc())
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        resp["id"] = rid
        print(json.dumps(resp), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
