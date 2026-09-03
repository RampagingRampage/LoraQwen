"""
record_voice_samples.py — pop-up mic recorder for gathering XTTS voice-
cloning samples.

Walks through a series of prompts designed to pull out a natural RANGE of
tone/pace/pitch — not scripts to read word-for-word, just things to talk
about naturally. Breathing, "ahem", pauses, laughs, "um"s — all good, don't
try to edit them out, that's exactly what makes a clone sound real instead
of robotic.

Run:
    python record_voice_samples.py [name]

Saves to voice_samples/<name>/<NN>_<label>.wav (44.1kHz mono 16-bit PCM).
"""

import os
import sys
import time
import wave
import threading
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, messagebox

SAMPLE_RATE = 44100
CHANNELS = 1
OUT_ROOT = "voice_samples"

# (label, prompt) — content matters less than the natural range of delivery
# each one is designed to pull out. Talk for ~10-20s on each, in your own
# words, at whatever pace it actually comes out.
PROMPTS = [
    ("casual_update",
     "Talk casually for a bit about what you're working on right now or did "
     "today — like texting a friend a quick update."),
    ("questions",
     "Ask a couple of genuine questions out loud — things you'd actually "
     "want the answer to, not made up ones."),
    ("excited",
     "Talk about something you're actually excited about right now — a "
     "project, a game, whatever — let yourself get into it, energy and all."),
    ("frustrated",
     "Vent a little about something mildly annoying — a bug, a bad UI, "
     "something that wasted your time. Keep it light, not heavy."),
    ("explaining",
     "Explain something technical like you're teaching a friend who doesn't "
     "know it yet — normal teaching pace, not rushed."),
    ("story",
     "Tell a short, real story — something that actually happened to you, "
     "start to finish."),
    ("joking",
     "Say something sarcastic, or joke around a little, the way you would "
     "with a friend."),
    ("reflective",
     "Talk slowly and thoughtfully for a bit. Pauses are fine — this one "
     "doesn't need energy."),
    ("directive",
     "Give a few short, direct commands or instructions, like you're telling "
     "someone exactly what to do, step by step."),
    ("laughing",
     "Say something that actually makes you laugh a little, and let the "
     "laugh happen for real if it does."),
]


def _dbfs(x):
    """int16-scale amplitude -> dBFS. x can be 0 (silence) -> returns a very
    negative number, never -inf, so callers can compare/format it safely."""
    return 20.0 * np.log10(max(float(x), 1.0) / 32768.0)


# Live-meter thresholds, in dBFS. Chosen from what actually happened the
# first time this tool was used without any of this: takes came out at -37
# to -45 dBFS RMS (see HANDOFF/session notes) -- consistently, uniformly
# quiet, which turned out to be the mic input gain itself being set low in
# Windows, not this tool doing anything wrong. This meter exists so that's
# visible WHILE recording instead of discovered after all 10 takes are done.
TOO_QUIET_DBFS  = -30.0   # below this: won't be salvageable without amplifying noise floor too
GOOD_LO_DBFS    = -24.0   # healthy conversational-speech RMS band
GOOD_HI_DBFS    = -12.0
CLIP_WARN_DBFS  = -3.0    # sustained level this loud risks real clipping, which normalization can't undo


class Recorder:
    def __init__(self):
        self.frames = []
        self.stream = None
        self.recording = False
        self.level = 0.0       # smoothed 0-1, for the meter bar width
        self.level_dbfs = -99.0  # smoothed dBFS, for the numeric readout
        self.peak_dbfs = -99.0   # running peak for this take, for the clip warning

    def _callback(self, indata, frames, time_info, status):
        self.frames.append(indata.copy())
        chunk = indata.astype(np.float32)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        peak = float(np.abs(chunk).max())
        dbfs = _dbfs(rms)
        self.level_dbfs = 0.7 * self.level_dbfs + 0.3 * dbfs
        self.level = 0.7 * self.level + 0.3 * min(1.0, rms / 32768.0 * 6)  # boosted for bar visibility
        self.peak_dbfs = max(self.peak_dbfs, _dbfs(peak))

    def start(self):
        self.frames = []
        self.recording = True
        self.peak_dbfs = -99.0
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if not self.frames:
            return None, 0.0
        audio = np.concatenate(self.frames, axis=0)
        duration = len(audio) / SAMPLE_RATE
        return audio, duration

    def take_quality(self, audio):
        """(overall_rms_dbfs, verdict) for the just-finished take, using the
        same percentile-based measure as the post-hoc normalizer -- robust to
        a single loud transient (mic bump/pop) that a plain peak check would
        be thrown off by."""
        arr = audio.astype(np.float64)
        rms_db = _dbfs(np.sqrt(np.mean(arr ** 2)))
        p999_db = _dbfs(np.percentile(np.abs(arr), 99.9))
        if rms_db < TOO_QUIET_DBFS:
            verdict = "quiet"
        elif p999_db > CLIP_WARN_DBFS:
            verdict = "loud"
        else:
            verdict = "good"
        return rms_db, verdict

    def save(self, audio, path, target_p999_dbfs=-6.0):
        """Auto-normalizes before writing: gain is computed from the 99.9th
        amplitude percentile rather than the true peak, so one loud transient
        (a mic bump/pop, not actual speech) can't anchor the whole take quiet
        the way a naive peak-normalize would -- same fix applied by hand to
        the very first batch of these that came out too quiet to use."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        arr = audio.astype(np.float64)
        p999 = np.percentile(np.abs(arr), 99.9)
        if p999 > 0:
            target = 32768 * (10 ** (target_p999_dbfs / 20))
            gain = target / p999
            arr = np.clip(arr * gain, -32768, 32767)
        out = arr.astype(np.int16)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # int16
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(out.tobytes())


class App(tk.Tk):
    def __init__(self, name):
        super().__init__()
        self.title("Voice Sample Recorder")
        self.geometry("620x460")
        self.configure(bg="#1e1e24")

        self.name = name
        self.out_dir = os.path.join(OUT_ROOT, name)
        self.idx = 0
        self.rec = Recorder()
        self.last_audio = None
        self.last_duration = 0.0
        self.done = [False] * len(PROMPTS)

        self._build_ui()
        self._refresh()
        self._poll_level()

    # ── UI ──────────────────────────────────────────────
    def _build_ui(self):
        fg = "#eaeaf0"
        bg = "#1e1e24"
        accent = "#7c9dff"

        header = tk.Label(self, text="Voice Sample Recorder", font=("Segoe UI", 16, "bold"),
                          fg=accent, bg=bg)
        header.pack(pady=(16, 4))

        self.progress_lbl = tk.Label(self, text="", font=("Segoe UI", 10), fg="#9a9ab0", bg=bg)
        self.progress_lbl.pack()

        self.label_lbl = tk.Label(self, text="", font=("Segoe UI", 11, "bold"), fg=accent, bg=bg)
        self.label_lbl.pack(pady=(14, 2))

        self.prompt_lbl = tk.Label(self, text="", font=("Segoe UI", 13), fg=fg, bg=bg,
                                   wraplength=560, justify="center")
        self.prompt_lbl.pack(pady=(4, 10), padx=20)

        hint = tk.Label(self, text="Breaths, pauses, \"um\"s, ahem, laughs — all good. Don't clean them out.",
                        font=("Segoe UI", 9, "italic"), fg="#7a7a8c", bg=bg)
        hint.pack(pady=(0, 14))

        # Level meter -- maps -60..0 dBFS onto the bar's width, with the
        # healthy target band shaded so there's an actual reference instead
        # of a bar that just goes up and down with no context for "enough".
        self.METER_W, self.METER_MIN_DB, self.METER_MAX_DB = 400, -60.0, 0.0
        self.meter = tk.Canvas(self, width=self.METER_W, height=18, bg="#111116", highlightthickness=0)
        self.meter.pack(pady=(0, 2))

        def _x(db):
            frac = (db - self.METER_MIN_DB) / (self.METER_MAX_DB - self.METER_MIN_DB)
            return max(0, min(self.METER_W, int(frac * self.METER_W)))

        self.meter.create_rectangle(_x(GOOD_LO_DBFS), 0, _x(GOOD_HI_DBFS), 18,
                                    fill="#25352a", width=0)  # target zone, drawn first (background)
        self.meter_bar = self.meter.create_rectangle(0, 0, 0, 18, fill=accent, width=0)
        self._meter_x = _x  # keep the mapping fn around for _poll_level

        self.level_lbl = tk.Label(self, text="", font=("Segoe UI", 9), fg="#7a7a8c", bg=bg)
        self.level_lbl.pack(pady=(0, 6))

        self.status_lbl = tk.Label(self, text="Ready", font=("Segoe UI", 10), fg="#9a9ab0", bg=bg)
        self.status_lbl.pack(pady=(0, 10))

        self.rec_btn = tk.Button(self, text="● Record", font=("Segoe UI", 13, "bold"),
                                 bg="#c94f4f", fg="white", activebackground="#e06565",
                                 width=16, height=2, relief="flat", command=self._toggle_record)
        self.rec_btn.pack(pady=(0, 12))

        row = tk.Frame(self, bg=bg)
        row.pack(pady=(0, 8))
        tk.Button(row, text="↺ Re-record", command=self._rerecord, width=14).grid(row=0, column=0, padx=6)
        tk.Button(row, text="◀ Prev", command=self._prev, width=10).grid(row=0, column=1, padx=6)
        tk.Button(row, text="Next ▶", command=self._next, width=10).grid(row=0, column=2, padx=6)
        tk.Button(row, text="Skip", command=self._skip, width=10).grid(row=0, column=3, padx=6)

        self.saved_lbl = tk.Label(self, text="", font=("Segoe UI", 9), fg="#6fbf73", bg=bg)
        self.saved_lbl.pack(pady=(10, 0))

        self.folder_lbl = tk.Label(self, text=f"Saving to: {os.path.abspath(self.out_dir)}",
                                   font=("Segoe UI", 8), fg="#5a5a6c", bg=bg)
        self.folder_lbl.pack(side="bottom", pady=8)

    def _refresh(self):
        label, prompt = PROMPTS[self.idx]
        n_done = sum(self.done)
        self.progress_lbl.config(text=f"Clip {self.idx + 1} of {len(PROMPTS)}   ·   {n_done} saved so far")
        self.label_lbl.config(text=label.replace("_", " ").upper() +
                              ("  ✓ saved" if self.done[self.idx] else ""))
        self.prompt_lbl.config(text=prompt)
        self.saved_lbl.config(text="")
        self.status_lbl.config(text="Ready")

    def _poll_level(self):
        if self.rec.recording:
            lvl = self.rec.level
            w = int(400 * lvl)
            db = self.rec.level_dbfs
            if db < TOO_QUIET_DBFS:
                color, msg = "#c94f4f", "TOO QUIET — get closer to the mic or raise input volume in Windows"
            elif db > CLIP_WARN_DBFS:
                color, msg = "#e0a030", "getting loud — back off slightly or lower input volume"
            else:
                color, msg = "#6fbf73", "good level"
            self.meter.coords(self.meter_bar, 0, 0, w, 18)
            self.meter.itemconfig(self.meter_bar, fill=color)
            self.level_lbl.config(text=f"{db:.0f} dBFS — {msg}", fg=color)
        else:
            self.meter.coords(self.meter_bar, 0, 0, 0, 18)
            self.level_lbl.config(text="")
        self.after(50, self._poll_level)

    # ── actions ─────────────────────────────────────────
    def _toggle_record(self):
        if not self.rec.recording:
            self.rec.start()
            self.rec_btn.config(text="■ Stop", bg="#4f4fc9")
            self.status_lbl.config(text="Recording…")
        else:
            audio, duration = self.rec.stop()
            self.rec_btn.config(text="● Record", bg="#c94f4f")
            if audio is None or duration < 1.0:
                self.status_lbl.config(text="Too short — try again")
                return
            self.last_audio = audio
            self.last_duration = duration
            self._save_current()

    def _save_current(self):
        label, _ = PROMPTS[self.idx]
        path = os.path.join(self.out_dir, f"{self.idx + 1:02d}_{label}.wav")
        rms_db, verdict = self.rec.take_quality(self.last_audio)
        self.rec.save(self.last_audio, path)  # saved file is auto-normalized regardless of verdict
        self.done[self.idx] = True
        self.saved_lbl.config(text=f"Saved — {self.last_duration:.1f}s, {rms_db:.0f} dBFS → {os.path.basename(path)}")
        if verdict == "quiet":
            # Auto-normalize still recovers this fine (that's the whole point),
            # but a take built almost entirely of noise floor + gain has worse
            # signal-to-noise than one that was actually spoken at a healthy
            # level -- worth knowing, not worth blocking on.
            self.status_lbl.config(text="Saved, but this take was quiet even before normalizing — "
                                        "consider re-recording closer to the mic.", fg="#e0a030")
        elif verdict == "loud":
            self.status_lbl.config(text="Saved, but this take clipped/was very loud — "
                                        "normalization can't undo distortion, consider re-recording.", fg="#e0a030")
        else:
            self.status_lbl.config(text="Saved. Re-record if you want another take, or move on.", fg="#9a9ab0")
        self.label_lbl.config(text=PROMPTS[self.idx][0].replace("_", " ").upper() + "  ✓ saved")
        self.progress_lbl.config(text=f"Clip {self.idx + 1} of {len(PROMPTS)}   ·   {sum(self.done)} saved so far")

    def _rerecord(self):
        self.done[self.idx] = False
        self._refresh()

    def _prev(self):
        if self.rec.recording:
            self._toggle_record()
        self.idx = (self.idx - 1) % len(PROMPTS)
        self._refresh()

    def _next(self):
        if self.rec.recording:
            self._toggle_record()
        if self.idx == len(PROMPTS) - 1 and all(self.done):
            messagebox.showinfo("Done", f"All {len(PROMPTS)} clips saved to:\n{os.path.abspath(self.out_dir)}\n\n"
                                        f"Pick your best 3-6 for cloning — you don't need to use all of them.")
        self.idx = (self.idx + 1) % len(PROMPTS)
        self._refresh()

    def _skip(self):
        self._next()


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "me"
    app = App(name)
    app.mainloop()
