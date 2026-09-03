"""
rank_responses.py — pop-up UI for collecting DPO preference data.

For each of N prompts (drawn from datasets/<name>.jsonl), generates 4
candidate responses from the trained character's own adapter (varied
temperature for real diversity) and lets you pick the best one — or write
your own if none of the 4 sound right. Saves incrementally so you can stop
and resume any time.

Requires the engine already running with <name>'s adapter loaded (start it
normally, same as chatting with the character — this tool doesn't manage
the engine itself, just talks to whatever's already up on port 8088).

Output:
    dpo_data/<name>_preferences.jsonl  — raw picks (prompt, all 4 candidates,
                                          which one/whether custom)
    dpo_data/<name>_dpo_pairs.jsonl    — derived {prompt, chosen, rejected}
                                          triples, ready for TRL's DPOTrainer

Run (always from the project root, not from inside tools/):
    python tools/rank_responses.py bob --rounds 50
"""

import os
import sys
import json
import glob
import random
import argparse
import threading
import tkinter as tk

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# This script lives in tools/ but imports root-level app modules (store,
# llama_backend) -- make sure the project root is importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tkinter import scrolledtext, messagebox

import store
from llama_backend import MANAGER

DPO_DIR = "dpo_data"
TEMPS = [0.7, 0.85, 1.0, 1.15]  # staggered for real diversity across the 4 candidates


def _find_char_id(name):
    """persona_forge names characters by their display name — find the id."""
    for c in store.list_characters():
        if c.get("name", "").lower() == name.lower():
            return c["id"]
    return None


def _already_ranked(name):
    """Prompts already saved in a previous session (any pick source, including
    skips-that-got-submitted) -- so re-running doesn't just reshuffle back over
    the same rounds you already did."""
    prefs_path = os.path.join(DPO_DIR, f"{name}_preferences.jsonl")
    if not os.path.exists(prefs_path):
        return set()
    seen = set()
    with open(prefs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["prompt"])
            except Exception:
                continue
    return seen


def _load_prompts(name, n):
    path = os.path.join("datasets", f"{name}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"No dataset at {path}")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    instructions = list({r["instruction"] for r in rows if r.get("instruction")})
    already = _already_ranked(name)
    fresh = [p for p in instructions if p not in already]
    random.shuffle(fresh)
    if already:
        print(f"{len(already)} prompt(s) already ranked in a previous session -- skipping those.")
    return fresh[:n]


def _system_prompt_for(char):
    persona = char.get("persona", "")
    name = char.get("name", "")
    return f"You are {name}. {persona}\nSpeak as {name}. No name prefix, no quotation marks."


class App(tk.Tk):
    def __init__(self, persona_name, char_id, prompts):
        super().__init__()
        self.title(f"Rank Responses — {persona_name}")
        self.geometry("900x700")
        bg = "#1e1e24"
        self.configure(bg=bg)

        self.persona_name = persona_name
        self.char_id = char_id
        self.prompts = prompts
        self.idx = 0
        self.candidates = []
        self.selected = tk.IntVar(value=-1)

        os.makedirs(DPO_DIR, exist_ok=True)
        self.prefs_path = os.path.join(DPO_DIR, f"{persona_name}_preferences.jsonl")
        self.pairs_path = os.path.join(DPO_DIR, f"{persona_name}_dpo_pairs.jsonl")
        self.saved_count = 0

        self._build_ui(bg)
        self._load_round()

    # ── UI ──────────────────────────────────────────────
    def _build_ui(self, bg):
        fg, accent = "#eaeaf0", "#7c9dff"

        top = tk.Frame(self, bg=bg)
        top.pack(fill="x", padx=16, pady=(14, 6))
        self.progress_lbl = tk.Label(top, text="", font=("Segoe UI", 10), fg="#9a9ab0", bg=bg)
        self.progress_lbl.pack(side="left")
        self.saved_lbl = tk.Label(top, text="", font=("Segoe UI", 10), fg="#6fbf73", bg=bg)
        self.saved_lbl.pack(side="right")

        self.prompt_box = tk.Label(self, text="", font=("Segoe UI", 12, "bold"), fg=accent,
                                   bg=bg, wraplength=850, justify="left", anchor="w")
        self.prompt_box.pack(fill="x", padx=16, pady=(0, 10))

        self.cand_frame = tk.Frame(self, bg=bg)
        self.cand_frame.pack(fill="both", expand=True, padx=16)
        self.cand_widgets = []
        for i in range(4):
            row = tk.Frame(self.cand_frame, bg="#2a2a33", highlightthickness=1,
                           highlightbackground="#3a3a45")
            row.pack(fill="x", pady=4)
            rb = tk.Radiobutton(row, text=f"#{i+1}", variable=self.selected, value=i,
                                bg="#2a2a33", fg=fg, selectcolor="#2a2a33",
                                activebackground="#2a2a33", font=("Segoe UI", 10, "bold"))
            rb.pack(side="left", anchor="n", padx=(6, 0), pady=6)
            txt = tk.Text(row, height=3, wrap="word", bg="#2a2a33", fg=fg,
                         font=("Segoe UI", 10), relief="flat", bd=0)
            txt.pack(side="left", fill="both", expand=True, padx=8, pady=6)
            txt.config(state="disabled")
            self.cand_widgets.append((rb, txt))

        custom_frame = tk.Frame(self, bg=bg)
        custom_frame.pack(fill="x", padx=16, pady=(10, 4))
        tk.Label(custom_frame, text="Or write your own instead:", font=("Segoe UI", 9, "italic"),
                fg="#9a9ab0", bg=bg).pack(anchor="w")
        self.custom_text = tk.Text(custom_frame, height=3, wrap="word", bg="#2a2a33", fg=fg,
                                   font=("Segoe UI", 10), relief="flat", bd=0,
                                   insertbackground=fg)
        self.custom_text.pack(fill="x", pady=(2, 0))

        btn_row = tk.Frame(self, bg=bg)
        btn_row.pack(pady=14)
        tk.Button(btn_row, text="↻ Regenerate 4", command=self._load_round, width=14).grid(row=0, column=0, padx=6)
        tk.Button(btn_row, text="Skip (no save)", command=self._skip, width=14).grid(row=0, column=1, padx=6)
        tk.Button(btn_row, text="✓ Submit & Next", command=self._submit, bg="#4f9dc9",
                 fg="white", font=("Segoe UI", 10, "bold"), width=18).grid(row=0, column=2, padx=6)

        self.status_lbl = tk.Label(self, text="", font=("Segoe UI", 9), fg="#9a9ab0", bg=bg)
        self.status_lbl.pack(pady=(0, 10))

    # ── round lifecycle ──────────────────────────────────
    def _load_round(self):
        if self.idx >= len(self.prompts):
            messagebox.showinfo("Done", f"All {len(self.prompts)} rounds complete.\n\n"
                                        f"{self.saved_count} rounds saved to:\n{self.prefs_path}")
            return
        prompt = self.prompts[self.idx]
        self.prompt_box.config(text=f"[{self.idx + 1}/{len(self.prompts)}]  {prompt}")
        self.progress_lbl.config(text=f"Round {self.idx + 1} of {len(self.prompts)}")
        self.custom_text.delete("1.0", "end")
        self.selected.set(-1)
        for rb, txt in self.cand_widgets:
            txt.config(state="normal")
            txt.delete("1.0", "end")
            txt.insert("1.0", "generating…")
            txt.config(state="disabled")
        self.status_lbl.config(text="Generating 4 candidates…")
        threading.Thread(target=self._generate_candidates, args=(prompt,), daemon=True).start()

    def _generate_candidates(self, prompt):
        char = store.get_character(self.char_id) or {}
        sys_prompt = _system_prompt_for(char)
        messages = [{"role": "system", "content": sys_prompt},
                   {"role": "user", "content": prompt}]
        results = []
        for t in TEMPS:
            try:
                text = MANAGER.complete(messages, char_id=self.char_id, temperature=t,
                                        max_tokens=150, enable_thinking=False)
            except Exception as e:
                text = f"[generation failed: {e}]"
            results.append(text.strip())
        self.candidates = results
        self.after(0, self._show_candidates)

    def _show_candidates(self):
        for (rb, txt), text in zip(self.cand_widgets, self.candidates):
            txt.config(state="normal")
            txt.delete("1.0", "end")
            txt.insert("1.0", text)
            txt.config(state="disabled")
        self.status_lbl.config(text="Pick the best one, or write your own below.")

    # ── actions ───────────────────────────────────────────
    def _submit(self):
        custom = self.custom_text.get("1.0", "end").strip()
        sel = self.selected.get()
        if not custom and sel == -1:
            messagebox.showwarning("Pick one", "Select a candidate or write your own response.")
            return
        prompt = self.prompts[self.idx]

        if custom:
            chosen = custom
            chosen_source = "custom"
            rejected_list = list(self.candidates)
        else:
            chosen = self.candidates[sel]
            chosen_source = f"candidate_{sel + 1}"
            rejected_list = [c for i, c in enumerate(self.candidates) if i != sel]

        record = {"prompt": prompt, "candidates": self.candidates,
                  "chosen": chosen, "chosen_source": chosen_source}
        with open(self.prefs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(self.pairs_path, "a", encoding="utf-8") as f:
            for rej in rejected_list:
                if rej and rej != chosen:
                    f.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rej},
                                       ensure_ascii=False) + "\n")

        self.saved_count += 1
        self.saved_lbl.config(text=f"{self.saved_count} saved")
        self.idx += 1
        self._load_round()

    def _skip(self):
        self.idx += 1
        self._load_round()


class _AttachedProc:
    """Stand-in for subprocess.Popen so MANAGER.is_ready() passes without this
    process having actually launched llama-server itself."""
    def poll(self):
        return None


def _attach_to_running_engine(flask_url="http://127.0.0.1:5000"):
    """This script runs in its own process, so the MANAGER singleton imported
    here never called .start() itself -- the real llama-server was started by
    server.py's Flask process, which owns the only accurate adapters/port
    state. Pull that state over HTTP so this MANAGER can talk to the same
    already-running engine instead of always seeing itself as 'not started'."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"{flask_url}/api/engine/status", timeout=5) as r:
            state = json.loads(r.read())
    except Exception:
        return False
    if state.get("status") != "ready":
        return False
    MANAGER.status = "ready"
    MANAGER.port = state.get("port", MANAGER.port)
    MANAGER.parallel = state.get("parallel", MANAGER.parallel)
    MANAGER.adapters = state.get("adapters", [])
    MANAGER.proc = _AttachedProc()
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("persona_name")
    ap.add_argument("--rounds", type=int, default=50)
    args = ap.parse_args()

    _attach_to_running_engine()
    if not MANAGER.is_ready():
        raise SystemExit(
            f"Engine isn't running. Start it with {args.persona_name}'s adapter loaded first "
            f"(same as chatting with the character normally), then run this again."
        )

    char_id = _find_char_id(args.persona_name)
    if not char_id:
        raise SystemExit(f"No character named '{args.persona_name}' found in characters/.")
    if MANAGER.adapter_id(char_id) is None:
        raise SystemExit(
            f"'{args.persona_name}' isn't among the adapters the running engine loaded. "
            f"Restart the engine so it includes this character's adapter, then try again."
        )

    prompts = _load_prompts(args.persona_name, args.rounds)
    if not prompts:
        raise SystemExit(f"No instructions found in datasets/{args.persona_name}.jsonl")
    print(f"{len(prompts)} rounds queued (from {args.rounds} requested, "
         f"capped by how many unique instructions exist).")

    app = App(args.persona_name, char_id, prompts)
    app.mainloop()


if __name__ == "__main__":
    main()
