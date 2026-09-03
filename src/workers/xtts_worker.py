"""
xtts_worker.py — persistent XTTS v2 synthesis worker.

Loads the model ONCE, then serves synthesis requests over stdin/stdout as
line-delimited JSON for the life of the process — so the caller (running in
lora_env, a different venv) doesn't pay XTTS's ~7s load cost per sentence,
only once. Same idea as llama-server: one long-lived process, many requests.

Must run under voice_env's python (XTTS needs an older transformers than the
training env uses):
    voice_env/Scripts/python.exe -u xtts_worker.py

Protocol — one JSON object per line, each direction:
  in:  {"text": "...", "ref_clips": ["path1.wav", ...], "out_path": "...", "language": "en"}
  out: {"ok": true, "path": "..."}  or  {"ok": false, "error": "..."}
First line out (before any requests) is {"ready": true} once the model has loaded.
"""

import os
import sys
import json
import time

os.environ["COQUI_TOS_AGREED"] = "1"  # auto-accept the Coqui Public Model License


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


_log("Loading XTTS v2...")
_t0 = time.time()
import torch
from TTS.api import TTS

_device = "cuda" if torch.cuda.is_available() else "cpu"
_tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(_device)
_log(f"Ready in {time.time() - _t0:.1f}s on {_device}")

print(json.dumps({"ready": True}), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        text = req["text"]
        ref_clips = req["ref_clips"]
        out_path = req["out_path"]
        language = req.get("language", "en")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        _tts.tts_to_file(text=text, speaker_wav=ref_clips, language=language, file_path=out_path)
        print(json.dumps({"ok": True, "path": out_path}), flush=True)
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}), flush=True)
