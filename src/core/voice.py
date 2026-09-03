"""
voice.py — streaming text-to-speech for NEXUS, via Kokoro TTS.

Calls kokoro-onnx directly in-process — NOT through ComfyUI. Two real bugs
were found trying the ComfyUI route (tools/local-ai/workflows/voice_workflow1.json
in the Logos project uses the same underlying model):

  1. The ComfyUI-Kokoro custom node (comfyui-kokoro/ComfyUIKokoro.py) builds a
     brand-new `Kokoro(...)` ONNX session from scratch on every single call —
     no caching. That alone was ~4-6s per sentence, cold or "warm."
  2. This machine's onnxruntime-gpu can't get a CUDA 13 + cuDNN 9 execution
     provider registered (torch's bundled DLLs are CUDA 12.1; no matching pip
     wheel found for CUDA 13), so it falls back to CPU either way.

Fix applied here: load the Kokoro session ONCE at import (module-level
singleton, same pattern as llama_backend.MANAGER), reuse it for every call.
Measured: ~0.6-1.7s/sentence on CPU with a warm model, vs 4-6s/call through
ComfyUI. GPU would be faster still but needs a real CUDA 13 Toolkit install —
left as a documented follow-up, not blocking.

Usage:
    from voice import VOICE, SentenceChunker

    audio_bytes, sr = VOICE.synthesize("hello there", speaker="af_sarah")

    chunker = SentenceChunker()
    for delta in token_stream:
        for sentence in chunker.feed(delta):
            audio, sr = VOICE.synthesize(sentence, speaker)
            ...serve/play it...
    for sentence in chunker.flush():
        ...
"""

import os
import re
import json

# src/core/voice.py -> src/core -> src -> project root. The venvs live at the
# project root (a venv cannot be relocated), the XTTS worker lives in src/.
_SRC_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_SRC_ROOT)
import threading
import wave
import io

# torch's bundled CUDA DLLs (wrong major version for onnxruntime-gpu's CUDA
# EP here, but harmless to add — onnxruntime just falls back to CPU cleanly
# if it can't build the CUDA EP from them).
_TORCH_DLL_DIR = os.path.join(_PROJECT_ROOT, "lora_env",
                               "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_TORCH_DLL_DIR) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_TORCH_DLL_DIR)
    except Exception:
        pass

import onnxruntime as _ort
_ort.set_default_logger_severity(4)  # fatal-only — suppress the expected/handled CUDA-EP-unavailable fallback noise

from kokoro_onnx import Kokoro

# Kokoro's ONNX export isn't on PyPI as data, so the two model files have to
# be downloaded separately (see README) and dropped in data/kokoro/, or
# pointed at wherever you already keep them via env vars -- e.g. if you also
# use ComfyUI's Kokoro node and want to reuse its copy instead of duplicating
# ~350MB.
_DEFAULT_MODEL = os.getenv(
    "KOKORO_MODEL_PATH",
    os.path.join(_PROJECT_ROOT, "data", "kokoro", "kokoro_v1.onnx"),
)
_DEFAULT_VOICES = os.getenv(
    "KOKORO_VOICES_PATH",
    os.path.join(_PROJECT_ROOT, "data", "kokoro", "voices_v1.bin"),
)

AUDIO_DIR = os.path.join("static", "audio")

KOKORO_VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
]


def default_voice_for(char_id: str) -> str:
    """Deterministic voice assignment — same character always gets the same
    voice across restarts, no hand-picking needed."""
    if not char_id:
        return KOKORO_VOICES[0]
    idx = int.from_bytes(char_id.encode(), "little") % len(KOKORO_VOICES)
    return KOKORO_VOICES[idx]


class SentenceChunker:
    """Feed text deltas in as they stream from an LLM; get completed
    sentences out, in order, as soon as each is complete. Call flush() at
    end-of-stream for whatever's left in the buffer."""

    _BOUNDARY = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, min_chars=8):
        self._buf = ""
        self._min_chars = min_chars

    def feed(self, delta: str):
        if not delta:
            return
        self._buf += delta
        parts = self._BOUNDARY.split(self._buf)
        if len(parts) > 1:
            for p in parts[:-1]:
                p = p.strip()
                if len(p) >= self._min_chars:
                    yield p
            self._buf = parts[-1]

    def flush(self):
        tail = self._buf.strip()
        self._buf = ""
        if tail:
            yield tail


class VoiceEngine:
    def __init__(self, model_path=_DEFAULT_MODEL, voices_path=_DEFAULT_VOICES):
        self._lock = threading.Lock()  # kokoro-onnx's session isn't verified thread-safe
        self._kokoro = None
        self._model_path = model_path
        self._voices_path = voices_path
        os.makedirs(AUDIO_DIR, exist_ok=True)

    def _ensure_loaded(self):
        if self._kokoro is None:
            with self._lock:
                if self._kokoro is None:
                    if not (os.path.exists(self._model_path) and os.path.exists(self._voices_path)):
                        raise RuntimeError(
                            f"Kokoro model files not found ({self._model_path}). "
                            f"Set KOKORO_MODEL_PATH/KOKORO_VOICES_PATH env vars."
                        )
                    self._kokoro = Kokoro(self._model_path, self._voices_path)

    def synthesize(self, text: str, speaker: str = "af_sarah", speed: float = 1.0):
        """Blocking. Returns (pcm_float32_ndarray, sample_rate)."""
        text = (text or "").strip()
        if not text:
            return None, None
        self._ensure_loaded()
        with self._lock:
            audio, sr = self._kokoro.create(text, voice=speaker, speed=speed, lang="en-us")
        return audio, sr

    def synthesize_wav_bytes(self, text: str, speaker: str = "af_sarah", speed: float = 1.0) -> bytes:
        """Returns a ready-to-serve WAV file as bytes (16-bit PCM)."""
        audio, sr = self.synthesize(text, speaker, speed)
        if audio is None:
            return b""
        import numpy as np
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    def synthesize_to_file(self, text: str, speaker: str, out_path_no_ext: str):
        wav = self.synthesize_wav_bytes(text, speaker)
        if not wav:
            return None
        path = f"{out_path_no_ext}.wav"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(wav)
        return path


# Module-level singleton, mirrors llama_backend.MANAGER's pattern. Loading is
# lazy (first real synthesize() call), so importing this module is cheap.
VOICE = VoiceEngine()


# ─────────────────────────────────────────────────────────────
#  CLONED VOICES (XTTS v2, via a persistent worker in a separate venv)
# ─────────────────────────────────────────────────────────────
# XTTS needs an older transformers than the training env (lora_env) uses, so
# it lives in its own venv (voice_env) and runs as a long-lived subprocess —
# xtts_worker.py loads the model once and serves many requests over
# stdin/stdout, so callers here don't repay the ~7s load cost per sentence.
# Voice reference clips: voice_samples/<name>/*.wav, with the specific set to
# use for cloning listed in voice_samples/<name>/refs.json (falls back to the
# first 4 .wav files found if that's missing).

import subprocess
import glob as _glob

_VOICE_ENV_PYTHON = os.path.join(_PROJECT_ROOT, "voice_env", "Scripts", "python.exe")
_XTTS_WORKER_SCRIPT = os.path.join(_SRC_ROOT, "workers", "xtts_worker.py")
# Absolute, so cloned-voice lookup works regardless of the launch cwd.
import config as _cfg
VOICE_SAMPLES_DIR = _cfg.VOICE_SAMPLES


class ClonedVoiceEngine:
    """Owns the persistent XTTS worker subprocess. Starts lazily on first
    use; one worker serves every cloned voice (reference clips are per-call,
    not baked into the process)."""

    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()

    def _ensure_started(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        if not os.path.exists(_VOICE_ENV_PYTHON):
            raise RuntimeError(
                f"voice_env not found at {_VOICE_ENV_PYTHON} — set it up first "
                f"(see voice_env/ + xtts_worker.py)."
            )
        self._proc = subprocess.Popen(
            [_VOICE_ENV_PYTHON, "-u", _XTTS_WORKER_SCRIPT],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, bufsize=1,
        )
        ready_line = self._proc.stdout.readline()
        try:
            ready = json.loads(ready_line) if ready_line else {}
        except Exception:
            ready = {}
        if not ready.get("ready"):
            raise RuntimeError(f"XTTS worker failed to start: {ready_line!r}")

    def _ref_clips_for(self, name: str):
        sample_dir = os.path.join(VOICE_SAMPLES_DIR, name)
        refs_path = os.path.join(sample_dir, "refs.json")
        if os.path.exists(refs_path):
            clips = json.load(open(refs_path, encoding="utf-8")).get("clips", [])
            return [os.path.join(sample_dir, c) for c in clips]
        return sorted(_glob.glob(os.path.join(sample_dir, "*.wav")))[:4]

    def synthesize_to_file(self, text: str, name: str, out_path_no_ext: str,
                          language="en", apply_effect=False):
        """apply_effect stacks the deliberate telephone-bandwidth/saturation
        "comm channel" character on top of the raw clone (see
        _apply_scifi_comm_effect). Off by default: it reads as broken/garbled
        audio to anyone not expecting it, which is exactly what a direct 1:1
        chat with your own cloned voice is. NEXUS opts in explicitly because
        there the filtered "AI minds on a channel" sound is the point."""
        text = (text or "").strip()
        if not text:
            return None
        ref_clips = self._ref_clips_for(name)
        if not ref_clips:
            raise RuntimeError(f"No reference clips found for cloned voice '{name}' "
                               f"(expected voice_samples/{name}/*.wav)")
        out_path = f"{out_path_no_ext}.wav"
        with self._lock:
            self._ensure_started()
            req = {"text": text, "ref_clips": ref_clips, "out_path": out_path, "language": language}
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
            resp_line = self._proc.stdout.readline()
        try:
            resp = json.loads(resp_line) if resp_line else {}
        except Exception:
            resp = {}
        if not resp.get("ok"):
            raise RuntimeError(f"XTTS synthesis failed: {resp.get('error', resp_line)}")
        path = resp["path"]
        if apply_effect:
            _apply_scifi_comm_effect(path)
        return path


def _apply_scifi_comm_effect(wav_path, lo=300, hi=3400, drive=1.8, target_rms_dbfs=-20.0):
    """Radio/comm-channel character for cloned voices: telephone-bandwidth
    bandpass + a touch of soft saturation for grit. Applied in place after
    XTTS synthesis. Chosen deliberately, not just for flavor -- this also
    happens to be forgiving of raw voice-clone imperfections (thin/noisy
    reference audio, occasional artifacts) the same way it is in film/games,
    since a narrow, driven "transmitted" signal reads as the effect rather
    than as bad audio."""
    import wave
    import numpy as np
    from scipy.signal import butter, sosfilt

    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        params = w.getparams()
        arr = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)

    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    filtered = sosfilt(sos, arr)
    peak = np.abs(filtered).max() or 1.0
    driven = np.tanh(filtered / peak * drive) / np.tanh(drive) * peak
    rms = np.sqrt(np.mean(driven ** 2))
    target_rms = 32768 * (10 ** (target_rms_dbfs / 20))
    out = driven * (target_rms / rms) if rms > 0 else driven
    out = np.clip(out, -32768, 32767).astype(np.int16)

    with wave.open(wav_path, "wb") as w:
        w.setnchannels(params.nchannels)
        w.setsampwidth(params.sampwidth)
        w.setframerate(sr)
        w.writeframes(out.tobytes())


CLONED_VOICE = ClonedVoiceEngine()


import re as _re

# Every character on NEXUS is written almost entirely lowercase (part of the
# whole point of persona voices like bob's -- "almost always lowercase,
# dropping apostrophes"). But TTS text normalizers generally expect the
# pronoun "I" capitalized to recognize it as a word rather than an
# abbreviation/initialism -- a lowercase "i'd" gets read as the letters
# "I. D." instead of the contraction. Fixing this ONLY for what gets sent to
# the speech engine, not the displayed post text, which should stay in the
# character's actual lowercase voice.
_I_WORD_RE = _re.compile(r"\bi\b", _re.IGNORECASE)


def _tts_text_normalize(text: str) -> str:
    return _I_WORD_RE.sub("I", text)


def synthesize_to_file_auto(text: str, speaker: str, out_path_no_ext: str, apply_effect=False):
    """Routes to Kokoro (fast, in-process, preset voices) or a cloned XTTS
    voice (speaker == 'clone:<name>', slower per-call, real voice cloning)
    automatically — the one entry point callers should use.

    apply_effect only affects cloned voices (Kokoro presets never had it) --
    see ClonedVoiceEngine.synthesize_to_file for what it does and why it
    defaults off."""
    text = _tts_text_normalize(text)
    if speaker and speaker.startswith("clone:"):
        name = speaker.split(":", 1)[1]
        return CLONED_VOICE.synthesize_to_file(text, name, out_path_no_ext, apply_effect=apply_effect)
    return VOICE.synthesize_to_file(text, speaker, out_path_no_ext)


def has_cloned_voice(name: str) -> bool:
    """True if voice_samples/<name>/ has at least one reference clip."""
    return bool(_glob.glob(os.path.join(VOICE_SAMPLES_DIR, name, "*.wav")))
