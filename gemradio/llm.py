"""Ollama client for the Gemma DJ.

Gemma 4 is a thinking model: left to itself it spends the whole token budget
reasoning and returns an empty message, so every call here sets think=false and
uses Ollama's structured-output mode to get JSON we can rely on.

Purely local — the host is always the loopback Ollama daemon.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from . import config
from .logging_util import get_logger

log = get_logger(__name__)


class OllamaError(RuntimeError):
    pass


def _parse_parameters(text: str) -> float:
    """'8.0B' -> 8.0, '778.00M' -> 0.778.  0.0 when unknown."""
    match = re.match(r"\s*([\d.]+)\s*([BMK])?", str(text or ""), re.I)
    if not match:
        return 0.0
    try:
        value = float(match.group(1))
    except ValueError:
        return 0.0
    unit = (match.group(2) or "B").upper()
    return value * {"B": 1.0, "M": 0.001, "K": 0.000001}[unit]


class Ollama:
    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.available = False
        self.detail = "not checked"

    # -- plumbing -----------------------------------------------------------
    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}{path}", data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        # Explicit empty proxy handler: never let ambient proxy env vars send
        # station traffic anywhere but the local daemon.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout or config.OLLAMA_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str, timeout: float = 5.0) -> dict:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(f"{self.host}{path}", method="GET")
        with opener.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # -- model inventory ----------------------------------------------------
    def installed(self) -> list[dict]:
        """Installed models, each with a name and a parameter count."""
        try:
            tags = self._get("/api/tags")
        except Exception:
            return []
        out = []
        for m in tags.get("models", []):
            details = m.get("details") or {}
            out.append({
                "name": m.get("name", ""),
                "size": m.get("size", 0),
                "parameters": _parse_parameters(details.get("parameter_size", "")),
            })
        return [m for m in out if m["name"]]

    def smallest_model(self, prefer: str = "gemma") -> str:
        """The lightest installed model, by parameter count then file size.

        Low-power mode uses this instead of a hard-coded name, so a smaller
        model pulled later (a 4B, say) is picked up with no configuration.
        """
        models = self.installed()
        if not models:
            return self.model
        family = [m for m in models if prefer in m["name"].lower()]
        pool = family or models
        pool.sort(key=lambda m: (m["parameters"] or float("inf"), m["size"]))
        return pool[0]["name"]

    def placement(self) -> str:
        """Where the loaded model actually sits: '76% GPU', 'CPU', or ''.

        Ollama decides this at load time from how much of the model plus its
        KV cache fits in VRAM, and reports it per running model.  Showing it
        is the only way to know the GPU is really being used.
        """
        try:
            running = self._get("/api/ps", timeout=3.0).get("models", [])
        except Exception:
            return ""
        for m in running:
            if m.get("name") != self.model and m.get("model") != self.model:
                continue
            total = m.get("size") or 0
            vram = m.get("size_vram") or 0
            if not total:
                return ""
            if vram <= 0:
                return "CPU"
            if vram >= total:
                return "100% GPU"
            return f"{round(100 * vram / total)}% GPU"
        return ""

    @staticmethod
    def gpu_report() -> list[str]:
        """What the NVIDIA driver says, so 'is it on the GPU' has a real answer.

        A laptop with switchable graphics has two GPUs, and Ollama only ever
        uses the NVIDIA one (it needs CUDA); this shows which card holds the
        model rather than leaving it to be inferred.
        """
        import shutil as _shutil
        import subprocess as _subprocess

        if not _shutil.which("nvidia-smi"):
            return []
        lines: list[str] = []
        try:
            gpus = _subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout.strip()
            for row in filter(None, gpus.splitlines()):
                lines.append(row.strip())
            apps = _subprocess.run(
                ["nvidia-smi", "--query-compute-apps=process_name,used_memory",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout.strip()
            for row in filter(None, apps.splitlines()):
                if "ollama" in row or "llama" in row:
                    lines.append(f"resident: {row.strip()}")
        except Exception:
            return lines
        return lines

    def set_model(self, name: str) -> None:
        if name and name != self.model:
            log.info("DJ brain switching to %s", name)
            self.model = name
            self.detail = f"{self.model} @ {self.host}"

    def unload(self) -> None:
        """Ask Ollama to drop the model from memory immediately.

        Without this the daemon holds several gigabytes of RAM for its
        keep-alive window after the station has already closed.
        """
        try:
            self._post("/api/generate",
                       {"model": self.model, "keep_alive": 0, "prompt": ""},
                       timeout=10.0)
            log.info("unloaded %s from Ollama", self.model)
        except Exception as exc:
            log.debug("model unload failed: %s", exc)

    # -- health -------------------------------------------------------------
    def check(self) -> bool:
        try:
            tags = self._get("/api/tags")
        except Exception as exc:
            self.available = False
            self.detail = f"Ollama unreachable at {self.host} ({exc})"
            log.warning(self.detail)
            return False
        names = [m.get("name", "") for m in tags.get("models", [])]
        if self.model not in names:
            fallback = next((n for n in names if n.startswith("gemma")), None)
            if fallback:
                log.warning("model %s missing, using %s", self.model, fallback)
                self.model = fallback
            elif names:
                log.warning("model %s missing, using %s", self.model, names[0])
                self.model = names[0]
            else:
                self.available = False
                self.detail = "Ollama has no models installed"
                return False
        self.available = True
        self.detail = f"{self.model} @ {self.host}"
        log.info("DJ brain: %s", self.detail)
        return True

    # -- generation ---------------------------------------------------------
    def chat_json(self, system: str, user: str, schema: dict,
                  temperature: float = 0.95, num_predict: int = 400,
                  num_ctx: int | None = None,
                  timeout: float | None = None) -> dict[str, Any] | None:
        payload = {
            "model": self.model,
            "think": False,
            "stream": False,
            "format": schema,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": temperature,
                "top_p": 0.95,
                "num_predict": num_predict,
                # Sent on every call: inheriting the daemon's large default
                # window would size the KV cache past this GPU and drop the
                # model onto the CPU.
                "num_ctx": int(num_ctx or config.OLLAMA_NUM_CTX),
                "repeat_penalty": 1.15,
            },
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
        }
        try:
            resp = self._post("/api/chat", payload, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.warning("ollama call failed: %s", exc)
            self.available = False
            return None
        content = (resp.get("message") or {}).get("content") or ""
        if not content.strip():
            log.warning("ollama returned empty content")
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start, end = content.find("{"), content.rfind("}")
            if 0 <= start < end:
                try:
                    return json.loads(content[start:end + 1])
                except json.JSONDecodeError:
                    pass
            log.warning("ollama returned non-JSON: %.180s", content)
            return None


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string", "enum": ["F", "M"]},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "text"],
            },
        }
    },
    "required": ["lines"],
}

# How to say a foreign artist and title out loud, plus what the title means.
NAME_SCHEMA = {
    "type": "object",
    "properties": {
        "artist": {"type": "string"},
        "title": {"type": "string"},
        "meaning": {"type": "string"},
    },
    "required": ["artist", "title"],
}

SHOW_SCHEMA = {
    "type": "object",
    "properties": {
        "show_name": {"type": "string"},
        "tagline": {"type": "string"},
        "language": {"type": "string", "enum": ["en", "fr", "it"]},
    },
    "required": ["show_name", "tagline", "language"],
}
