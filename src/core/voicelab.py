"""
core/voicelab.py — voice sample management for the Voice tab.

The recording UI moved from a Tkinter window into the browser, so this module
owns what that Tk app used to: the guided prompt list, where clips land, which
clips XTTS actually uses as references, and per-clip level analysis so you can
see a clip is too quiet or clipped before you find out from the clone.

Browsers hand us WebM/Ogg from MediaRecorder, so uploads are transcoded to the
44.1 kHz mono 16-bit PCM WAV that XTTS expects — via ffmpeg when it's on PATH,
and otherwise by decoding in-process.
"""

import os
import json
import wave
import shutil
import struct
import subprocess

import config

# The prompts are the point of the recorder: they pull a natural RANGE of tone
# and pace out of you. Not scripts to read — things to talk about.
PROMPTS = [
    ("casual_update",
     "Talk casually for a bit about what you're working on right now or did "
     "today — like texting a friend a quick update."),
    ("questions",
     "Ask a couple of genuine questions out loud — things you'd actually want "
     "the answer to, not made up ones."),
    ("excited",
     "Talk about something you're actually excited about right now — a project, "
     "a game, whatever — let yourself get into it, energy and all."),
    ("frustrated",
     "Vent a little about something mildly annoying — a bug, a bad UI, something "
     "that wasted your time. Keep it light, not heavy."),
    ("explaining",
     "Explain something technical like you're teaching a friend who doesn't know "
     "it yet — normal teaching pace, not rushed."),
    ("story",
     "Tell a short, real story — something that actually happened to you, start "
     "to finish."),
    ("joking",
     "Say something sarcastic, or joke around a little, the way you would with a "
     "friend."),
    ("reflective",
     "Talk slowly and thoughtfully for a bit. Pauses are fine — this one doesn't "
     "need energy."),
    ("directive",
     "Give a few short, direct commands or instructions, like you're telling "
     "someone exactly what to do, step by step."),
    ("laughing",
     "Say something that actually makes you laugh a little, and let the laugh "
     "happen for real if it does."),
]

TARGET_RATE = 44100
GOOD_MIN_SECONDS = 6
GOOD_MAX_SECONDS = 45


def prompts():
    return [{"index": i + 1, "label": lbl, "prompt": txt,
             "filename": f"{i+1:02d}_{lbl}.wav"}
            for i, (lbl, txt) in enumerate(PROMPTS)]


def _dir(name):
    safe = os.path.basename(str(name or "")).strip()
    if not safe:
        raise ValueError("A voice name is required")
    d = os.path.join(config.VOICE_SAMPLES, safe)
    os.makedirs(d, exist_ok=True)
    return d


def list_voices():
    """Every folder under voice_samples/ that has at least one clip in it."""
    out = []
    if not os.path.isdir(config.VOICE_SAMPLES):
        return out
    for entry in sorted(os.listdir(config.VOICE_SAMPLES)):
        d = os.path.join(config.VOICE_SAMPLES, entry)
        if not os.path.isdir(d):
            continue
        clips = [f for f in os.listdir(d) if f.lower().endswith(".wav")]
        if clips:
            out.append({"name": entry, "clips": len(clips),
                        "refs": len(_refs(entry))})
    return out


def _refs_path(name):
    return os.path.join(_dir(name), "refs.json")


def _refs(name):
    p = _refs_path(name)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f).get("clips", []) or []
        except (OSError, json.JSONDecodeError):
            return []
    return []


def set_refs(name, clips):
    """Choose which clips XTTS clones from. Four or five varied ones beat all
    ten — more isn't better, variety is."""
    d = _dir(name)
    have = {f for f in os.listdir(d) if f.lower().endswith(".wav")}
    keep = [os.path.basename(c) for c in clips if os.path.basename(c) in have]
    tmp = _refs_path(name) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"clips": keep}, f, indent=1)
    os.replace(tmp, _refs_path(name))
    return keep


def analyze_clip(path):
    """Duration plus peak/RMS in dBFS, so the UI can say 'too quiet' or
    'clipping' instead of leaving you to guess from a waveform."""
    try:
        with wave.open(path, "rb") as w:
            n, rate, width, chans = (w.getnframes(), w.getframerate(),
                                     w.getsampwidth(), w.getnchannels())
            raw = w.readframes(n)
    except (wave.Error, OSError) as e:
        return {"error": str(e)}

    seconds = n / float(rate or 1)
    peak = rms = 0.0
    if width == 2 and raw:
        try:
            import numpy as np
            arr = np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0
            if chans > 1:
                arr = arr.reshape(-1, chans).mean(axis=1)
            peak = float(abs(arr).max()) if arr.size else 0.0
            rms = float((arr ** 2).mean() ** 0.5) if arr.size else 0.0
        except ImportError:
            count = len(raw) // 2
            vals = struct.unpack(f"<{count}h", raw[:count * 2])
            peak = max(abs(v) for v in vals) / 32768.0 if vals else 0.0
            rms = (sum(v * v for v in vals) / max(count, 1)) ** 0.5 / 32768.0

    def db(x):
        return round(20 * (x and __import__("math").log10(x) or -5), 1) if x > 0 else -99.0

    notes = []
    if seconds < GOOD_MIN_SECONDS:
        notes.append("short — aim for 10-30s")
    elif seconds > GOOD_MAX_SECONDS:
        notes.append("long — XTTS only needs a slice")
    if peak >= 0.99:
        notes.append("clipping")
    elif db(rms) < -34:
        notes.append("quiet — move closer to the mic")

    return {
        "seconds": round(seconds, 2), "rate": rate, "channels": chans,
        "peak_db": db(peak), "rms_db": db(rms), "notes": notes,
        "ok": not notes,
    }


def list_clips(name):
    d = _dir(name)
    refs = set(_refs(name))
    out = []
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith(".wav"):
            continue
        p = os.path.join(d, f)
        entry = {"file": f, "bytes": os.path.getsize(p),
                 "modified": os.path.getmtime(p),
                 "is_ref": f in refs or not refs,
                 "url": f"/api/voice/clip/{os.path.basename(name)}/{f}"}
        entry.update(analyze_clip(p))
        out.append(entry)
    return out


def delete_clip(name, filename):
    d = _dir(name)
    f = os.path.basename(filename)
    p = os.path.join(d, f)
    if not os.path.exists(p):
        return False
    os.remove(p)
    refs = [c for c in _refs(name) if c != f]
    if refs != _refs(name):
        set_refs(name, refs)
    return True


# ─────────────────────────────────────────────────────────────
#  UPLOAD / TRANSCODE
# ─────────────────────────────────────────────────────────────

def _ffmpeg():
    return shutil.which("ffmpeg")


def save_upload(name, filename, data, source_ext=".webm"):
    """Persist a recorded or uploaded clip as 44.1k mono 16-bit PCM WAV."""
    d = _dir(name)
    stem = os.path.splitext(os.path.basename(filename))[0] or "clip"
    dest = os.path.join(d, f"{stem}.wav")

    # Already a WAV in the right shape? Just normalize it through the same path
    # so channel count and rate are guaranteed.
    tmp_src = os.path.join(d, f".upload_{stem}{source_ext}")
    with open(tmp_src, "wb") as f:
        f.write(data)

    try:
        ff = _ffmpeg()
        if ff:
            proc = subprocess.run(
                [ff, "-y", "-hide_banner", "-loglevel", "error",
                 "-i", tmp_src, "-ac", "1", "-ar", str(TARGET_RATE),
                 "-c:a", "pcm_s16le", dest],
                capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[:300]}")
        else:
            _decode_without_ffmpeg(tmp_src, dest, source_ext)
    finally:
        try:
            os.remove(tmp_src)
        except OSError:
            pass

    info = analyze_clip(dest)
    return {"file": os.path.basename(dest), **info}


def _decode_without_ffmpeg(src, dest, ext):
    """Fallback when ffmpeg isn't on PATH.

    Plain WAV is resampled in-process. Compressed browser formats (WebM/Ogg)
    genuinely need a decoder, so say so clearly rather than writing a file that
    XTTS will choke on later.
    """
    if ext.lower() in (".wav", ".wave"):
        _resample_wav(src, dest)
        return
    try:
        import soundfile as sf
        import numpy as np
        audio, rate = sf.read(src, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if rate != TARGET_RATE:
            idx = np.linspace(0, len(mono) - 1, int(len(mono) * TARGET_RATE / rate))
            mono = np.interp(idx, np.arange(len(mono)), mono)
        pcm = (np.clip(mono, -1, 1) * 32767).astype("int16")
        with wave.open(dest, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(TARGET_RATE)
            w.writeframes(pcm.tobytes())
    except Exception as e:
        raise RuntimeError(
            "Recorded audio needs ffmpeg (or soundfile) to decode, and neither "
            f"could handle it here: {e}. Install ffmpeg and add it to PATH, or "
            "upload a .wav instead."
        )


def _resample_wav(src, dest):
    import numpy as np
    with wave.open(src, "rb") as w:
        rate, chans, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise RuntimeError(f"Only 16-bit WAV is supported here (got {width*8}-bit).")
    arr = np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0
    if chans > 1:
        arr = arr.reshape(-1, chans).mean(axis=1)
    if rate != TARGET_RATE:
        idx = np.linspace(0, len(arr) - 1, int(len(arr) * TARGET_RATE / rate))
        arr = np.interp(idx, np.arange(len(arr)), arr)
    pcm = (np.clip(arr, -1, 1) * 32767).astype("int16")
    with wave.open(dest, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_RATE)
        w.writeframes(pcm.tobytes())
