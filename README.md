# LoraQwen Forge

Turns a folder of text into a LoRA-tuned character with a cloned voice, and
gives you a browser app to build, train, evaluate, and talk to it. One Flask
backend, ten tabs, nothing in the terminal.

```powershell
lora_env\Scripts\python src\app.py
```

Then open <http://localhost:5000>. First time on a machine, see
[Setup](#setup) below.

> No `.bat` launchers are included in this repo — Windows tags `.bat` files
> downloaded from GitHub (as a ZIP, or sometimes via clone) with the
> Mark-of-the-Web, which trips SmartScreen and script-execution warnings. The
> commands below are what those scripts would have run; paste any of them
> into your own `run.bat` / `install.bat` if you want the double-click
> convenience back — they're plain, harmless commands, Windows just doesn't
> trust `.bat` files it didn't create itself.

---

## What it does

Point it at a folder of documents for a persona — your own writing, a
character's dialogue, a public-domain author, anything with a consistent
voice. It distills that into a voice profile, generates synthetic training
data in that voice (supervised as closely as you want), lets you clean what
came out, trains a QLoRA adapter, and checks whether the training actually
worked against the source, not just against loss going down. A chat tab
talks to the result.

Voice is a separate, optional track: record and clone anyone's voice and
attach it to any character — it doesn't have to be the same person the text
persona was built from.

## The tabs

| Tab | What it's for |
|---|---|
| **Inference** | Chat. Multiple conversations, per-chat system prompt, temperature, token limit, tools and memory injection. |
| **Feed** | NEXUS — characters think privately, post publicly, and speak aloud. |
| **Characters** | Create and edit characters; browse and prune their memories. |
| **Persona** | Upload raw writing, filter it, distil a voice profile, hand-edit the result. |
| **Data** | Generate samples — from the local engine, LM Studio, or a Claude Code handshake (see below) — with the supervision slider, review the held ones, clean the dataset, rank responses for DPO. |
| **Train** | Every hyperparameter, three presets, live train/eval loss curve. |
| **Evaluate** | Side-by-side generation, style scorecard, memorization check, regression set. |
| **Voice** | Record in the browser, pick reference clips, preview the 28 built-in voices. |
| **Engine** | Start/stop llama-server, export GGUFs, download models, kill orphans. |
| **About / Help** | The full nine-step walkthrough, and troubleshooting for the things that actually go wrong. |

## Dataset generation sources

The Data tab can generate training examples three ways:

- **Local engine** — the llama-server this app manages, generating directly.
- **LM Studio** (or any OpenAI-compatible endpoint) — point `LMSTUDIO_URL` in
  `.env` at it.
- **Claude Code handshake** — no API key and no network call from this app at
  all. Choosing it writes a self-contained request file (the persona profile,
  topic list, and instructions) to `data/datasets/_requests/<name>.json`. You
  run [Claude Code](https://claude.com/claude-code) against that file
  yourself — Claude Code reads it, writes JSONL answers back to
  `data/datasets/_responses/<name>.jsonl`, and the Data tab's "Collect Claude
  output" button imports them through the same dedupe and supervision rules
  as a live run. It's just a plain JSON file on disk; anything you can point
  at a file works, Claude Code is simply what this was built against.

## Setup

```powershell
# 1. Two virtualenvs -- training and voice cloning need different
#    transformers versions, so they're kept separate.
python -m venv lora_env
python -m venv voice_env
lora_env\Scripts\python -m pip install --upgrade pip
lora_env\Scripts\python -m pip install -r requirements.txt
voice_env\Scripts\python -m pip install --upgrade pip
voice_env\Scripts\python -m pip install -r requirements-voice.txt

# 2. Config -- copy the template, then edit .env if you want different
#    ports, paths or defaults. Nothing is hardcoded in the source.
copy .env.example .env

# 3. Inference engine -- download a CUDA release of llama.cpp from
#    https://github.com/ggml-org/llama.cpp/releases and extract it into
#    runtime\llama.cpp\. The prebuilt zip omits three files the app also
#    needs for GGUF export -- copy these from the llama.cpp source repo:
#      runtime\llama.cpp\convert_hf_to_gguf.py
#      runtime\llama.cpp\convert_lora_to_gguf.py
#      runtime\llama.cpp\gguf-py\   (whole folder)

# 4. Verify everything is in place
lora_env\Scripts\python src\verify.py

# 5. Run
lora_env\Scripts\python src\app.py
```

The base model (Qwen3-8B, ~16 GB) downloads from the app's Engine tab the
first time you need it — no separate step required.

The 28 built-in Kokoro voices (Voice tab preview, and the Feed's spoken
posts) need two model files that aren't on PyPI: download `kokoro_v1.onnx`
and `voices_v1.bin` from the [kokoro-onnx
releases](https://github.com/thewh1teagle/kokoro-onnx) and drop them in
`data/kokoro/` (or point `KOKORO_MODEL_PATH`/`KOKORO_VOICES_PATH` in `.env`
at wherever you already have them). Everything else — chat, persona,
training, evaluation, voice *cloning* via XTTS — works without them.

**Stopping:** closing the app window does not stop `llama-server.exe`, which
Windows keeps running after its parent process exits. Use the Engine tab's
"Kill orphans", or:

```powershell
taskkill /F /IM llama-server.exe /T
```

### Linux / macOS (untested)

This project has only ever been built and run on Windows with an NVIDIA GPU
— nothing below has actually been tried. The app itself is plain Flask and
should be platform-agnostic, but two Windows-specific things stand in the
way of a clean port:

- `requirements.txt` pins `torch==2.5.1+cu121` — a CUDA 12.1 wheel from
  PyTorch's own index, not PyPI. On Linux with an NVIDIA GPU this should
  install fine as-is (maybe swapping the CUDA version to match your
  drivers); on macOS there's no CUDA, so you'd need to swap in the CPU/MPS
  torch build yourself.
- QLoRA training uses `bitsandbytes`, which needs CUDA. That means the
  **Train tab will not work on Apple Silicon** — there's no NVIDIA GPU to
  target. Chat/inference against a model someone already trained should
  still work via a Metal build of llama.cpp, independent of the training
  stack.

If you try it, the command shapes carry over directly — swap the venv's
`Scripts\python.exe` for `bin/python`, and download the matching
Linux/macOS `llama.cpp` release instead of the Windows CUDA one:

```bash
python3.11 -m venv lora_env
python3.11 -m venv voice_env
lora_env/bin/python -m pip install --upgrade pip
lora_env/bin/python -m pip install -r requirements.txt
voice_env/bin/python -m pip install --upgrade pip
voice_env/bin/python -m pip install -r requirements-voice.txt

cp .env.example .env

# download a matching llama.cpp release from
# https://github.com/ggml-org/llama.cpp/releases and extract it into
# runtime/llama.cpp/, same three extra conversion files as above (the
# binary is just "llama-server", no .exe)

lora_env/bin/python src/verify.py
lora_env/bin/python src/app.py
```

Stopping the engine: `pkill -f llama-server` instead of `taskkill`.

Please open an issue (or a PR) with what did and didn't work if you get this
running on either platform.

## Layout

```
LoraQwen/
├─ README.md                                     docs
├─ .env  .env.example  requirements*.txt         config
│
├─ src/            the application
│   ├─ app.py          one backend, ~92 routes
│   ├─ core/           engine, store, voice, pipeline, persona, generate,
│   │                  dataset, evaluate, feed, voicelab, config
│   ├─ workers/        train_worker.py, xtts_worker.py (subprocess isolation)
│   └─ static/         index.html + css + one JS module per tab
│
├─ data/           everything you make
│   ├─ training_input/   raw documents and distilled voice profiles
│   ├─ datasets/         training data, review queues, snapshots
│   ├─ loras/            trained adapters
│   ├─ gguf_output/      base model + exported adapters
│   ├─ characters/       configs, chats, shared RAG store
│   ├─ voice_samples/    recorded clips + reference selection
│   ├─ dpo_data/         preference picks and derived pairs
│   ├─ models/           downloaded HuggingFace models
│   └─ evals/            regression prompts and past runs
│
├─ runtime/        third-party binaries and scratch state
│   ├─ llama.cpp/        the inference engine and conversion scripts
│   ├─ .train_ipc/       parent↔worker training status protocol
│   └─ .hf_temp_cache/   HuggingFace download cache
│
└─ lora_env/  voice_env/    virtualenvs — see note below
```

**The venvs stay at the root.** A virtualenv bakes its own absolute path into
`pyvenv.cfg` and every launcher stub in `Scripts/`, so moving one breaks it.
XTTS needs an older `transformers` than training does, which is why there are
two.

## Configuration

Every port, path and default is in `.env` — nothing is hardcoded in the source.
Directory names there resolve against `data/` (content) and `runtime/`
(binaries and scratch) automatically; give an absolute path to put something on
another drive.

## Requirements

- Windows, Python 3.11, an NVIDIA GPU with CUDA — the only combination this
  has actually been built and tested on. Linux is likely fine with the same
  GPU; macOS can chat/infer but can't train (see
  [Linux / macOS](#linux--macos-untested)).
- ~35 GB free: 16 GB base model, 8 GB GGUF, 6 GB training venv
- `runtime/llama.cpp/` with `llama-server.exe` **and** three files the prebuilt
  release omits: `convert_hf_to_gguf.py`, `convert_lora_to_gguf.py`, and the
  `gguf-py/` package. `src/verify.py` checks for all of them (see
  [Setup](#setup)).

## Notes worth knowing

- **Stop the engine before training.** Two processes on one GPU is the most
  common out-of-memory failure. A "Stop" button lives in the top bar on every
  tab so you don't have to go find the Engine tab first.
- **`llama-server.exe` outlives its parent on Windows.** Use the Engine tab's
  "Kill orphans", or `taskkill /F /IM llama-server.exe /T` (see
  [Setup](#setup)).
- **Set an eval split.** It is the only signal that tells you the model is
  memorizing rather than learning, and at a few hundred examples that happens
  early.
- **`pyarrow` must import before `torch`.** `transformers.Trainer` pulls it in
  transitively and the reverse order segfaults on Windows. The pipeline handles
  this; new training entry points need to do the same.

The Help tab (in the app) covers everything else.

## Privacy

Nothing you make with this app is in this repository. `data/`, `runtime/`,
`lora_env/`, `voice_env/`, and `.env` are all gitignored — your characters,
chat logs, datasets, DPO pairs, trained LoRA adapters, downloaded base models,
and voice recordings stay local to your machine. The only tracked file under
`data/` is `data/training_input/README.md`, which is just usage
instructions.
