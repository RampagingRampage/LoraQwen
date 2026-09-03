"""
core/config.py — one place where every path, port and default lives.

Everything is read from .env (or the process environment) with a sane
fallback, so nothing in the app has to hardcode a port or a directory
name again. Values are resolved ONCE at import against PROJECT_ROOT, so
every module agrees on where things are regardless of the cwd it was
started from.
"""

import os
from pathlib import Path

# src/core/config.py -> src/core -> src -> project root
SRC_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_ROOT)

# The project is laid out in four top-level areas:
#   src/      the application (code + static assets)
#   data/     everything you make: personas, datasets, adapters, voices
#   runtime/  third-party binaries and scratch state (llama.cpp, caches, IPC)
#   root      launchers, docs and config only
DATA_ROOT    = os.path.join(PROJECT_ROOT, "data")
RUNTIME_ROOT = os.path.join(PROJECT_ROOT, "runtime")


def _load_dotenv(path=None):
    """Minimal .env reader — no dependency, no surprises. KEY=value lines,
    '#' comments, optional surrounding quotes. Never overwrites a variable
    that is already set in the real environment."""
    path = path or os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            os.environ.setdefault(key, val)


_load_dotenv()


def _s(key, default):
    return os.environ.get(key, default)


def _i(key, default):
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _f(key, default):
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _b(key, default):
    return str(os.environ.get(key, str(default))).strip().lower() in ("1", "true", "yes", "on")


def _path(key, default_rel, base=None):
    """Resolve a configured path. Relative values are anchored to the given
    base (the project root by default) so the app works the same whether it was
    launched from the root, from a shortcut with a different working directory,
    or from a subfolder."""
    v = os.environ.get(key, default_rel)
    return v if os.path.isabs(v) else os.path.join(base or PROJECT_ROOT, v)


# ── ports / hosts ────────────────────────────────────────────
WEB_HOST      = _s("WEB_HOST", "127.0.0.1")
WEB_PORT      = _i("WEB_PORT", 5000)
ENGINE_PORT   = _i("ENGINE_PORT", 8088)
ENGINE_URL    = _s("ENGINE_URL", f"http://127.0.0.1:{ENGINE_PORT}")
LMSTUDIO_URL  = _s("LMSTUDIO_URL", "http://localhost:1234")

# ── directories ──────────────────────────────────────────────
# ── data/  — everything you make ─────────────────────────
CHARACTERS_DIR   = _path("CHARACTERS_DIR", "characters", DATA_ROOT)
DATASETS_DIR     = _path("DATASETS_DIR", "datasets", DATA_ROOT)
DPO_DIR          = _path("DPO_DIR", "dpo_data", DATA_ROOT)
LORAS_DIR        = _path("LORAS_DIR", "loras", DATA_ROOT)
GGUF_DIR         = _path("GGUF_DIR", "gguf_output", DATA_ROOT)
MODELS_DIR       = _path("MODELS_DIR", "models", DATA_ROOT)
TRAINING_INPUT   = _path("TRAINING_INPUT", "training_input", DATA_ROOT)
VOICE_SAMPLES    = _path("VOICE_SAMPLES", "voice_samples", DATA_ROOT)
EVAL_DIR         = _path("EVAL_DIR", "evals", DATA_ROOT)
REVIEW_DIR       = os.path.join(DATASETS_DIR, "_review")
GEN_REQUESTS     = os.path.join(DATASETS_DIR, "_requests")
GEN_RESPONSES    = os.path.join(DATASETS_DIR, "_responses")

# ── runtime/  — third-party binaries and scratch state ───
LLAMA_CPP_DIR    = _path("LLAMA_CPP_DIR", "llama.cpp", RUNTIME_ROOT)
IPC_DIR          = _path("IPC_DIR", ".train_ipc", RUNTIME_ROOT)
HF_HOME          = _path("HF_HOME", ".hf_temp_cache", RUNTIME_ROOT)

# ── src/  — the application itself ───────────────────────
STATIC_DIR       = _path("STATIC_DIR", "static", SRC_ROOT)
AUDIO_DIR        = os.path.join(STATIC_DIR, "audio")
WORKERS_DIR      = os.path.join(SRC_ROOT, "workers")

# ── root  — the virtualenvs stay put ─────────────────────
# A venv bakes its own absolute path into pyvenv.cfg and the launcher stubs in
# Scripts/, so these cannot be relocated without rebuilding them.
LORA_ENV         = _path("LORA_ENV", "lora_env")
VOICE_ENV        = _path("VOICE_ENV", "voice_env")

# ── engine defaults ──────────────────────────────────────────
DEFAULT_CTX      = _i("DEFAULT_CTX", 8192)
DEFAULT_NGL      = _i("DEFAULT_NGL", 999)
DEFAULT_PARALLEL = _i("DEFAULT_PARALLEL", 2)
DEFAULT_MODEL    = _s("DEFAULT_MODEL", "Qwen/Qwen3-8B")

# ── inference defaults (the Chat tab's starting values) ──────
CHAT_TEMPERATURE = _f("CHAT_TEMPERATURE", 0.7)
CHAT_MAX_TOKENS  = _i("CHAT_MAX_TOKENS", 512)
CHAT_TOP_P       = _f("CHAT_TOP_P", 0.95)

# ── feed defaults ────────────────────────────────────────────
FEED_MAX_POSTS    = _i("FEED_MAX_POSTS", 200)
FEED_THINK_TEMP   = _f("FEED_THINK_TEMP", 0.9)
FEED_POST_TEMP    = _f("FEED_POST_TEMP", 0.85)
FEED_THINK_TOKENS = _i("FEED_THINK_TOKENS", 120)
FEED_POST_TOKENS  = _i("FEED_POST_TOKENS", 150)
FEED_AUTO_MEMORIZE = _b("FEED_AUTO_MEMORIZE", False)   # off by default now — see audit finding 01
FEED_MEMORY_CAP    = _i("FEED_MEMORY_CAP", 60)

# ── generation defaults ──────────────────────────────────────
GEN_TOTAL       = _i("GEN_TOTAL", 300)
GEN_BATCH       = _i("GEN_BATCH", 8)
GEN_TEMPERATURE = _f("GEN_TEMPERATURE", 0.9)
GEN_TIMEOUT_S   = _i("GEN_TIMEOUT_S", 240)
GEN_DEDUPE      = _f("GEN_DEDUPE", 0.62)
GEN_SUPERVISE_PCT = _i("GEN_SUPERVISE_PCT", 0)

# ── training defaults ────────────────────────────────────────
TRAIN_DEFAULTS = {
    "model_name":     DEFAULT_MODEL,
    "max_seq_length": _i("TRAIN_SEQ_LEN", 1024),
    "lora_rank":      _i("TRAIN_RANK", 16),
    "lora_alpha":     _i("TRAIN_ALPHA", 32),
    "lora_dropout":   _f("TRAIN_DROPOUT", 0.05),
    "num_epochs":     _i("TRAIN_EPOCHS", 3),
    "batch_size":     _i("TRAIN_BATCH", 2),
    "grad_accum":     _i("TRAIN_GRAD_ACCUM", 4),
    "learning_rate":  _f("TRAIN_LR", 1.5e-4),
    "warmup_ratio":   _f("TRAIN_WARMUP", 0.03),
    "lr_scheduler":   _s("TRAIN_SCHEDULER", "cosine"),
    "logging_steps":  _i("TRAIN_LOG_STEPS", 5),
    "save_steps":     _i("TRAIN_SAVE_STEPS", 200),
    "eval_split_pct": _i("TRAIN_EVAL_SPLIT", 10),
    "seed":           _i("TRAIN_SEED", 42),
    "target_modules": _s("TRAIN_TARGETS", "q_proj,k_proj,v_proj,o_proj"),
}

TRAIN_PRESETS = {
    "quick":    {"lora_rank": 8,  "lora_alpha": 16, "num_epochs": 2,
                 "label": "Quick",    "note": "Sanity pass on a new dataset. ~10 min."},
    "standard": {"lora_rank": 16, "lora_alpha": 32, "num_epochs": 3,
                 "label": "Standard", "note": "Known-good baseline. Bob was trained this way."},
    "deep":     {"lora_rank": 32, "lora_alpha": 64, "num_epochs": 3,
                 "label": "Deep",     "note": "For datasets past ~800 examples. More VRAM."},
}


def ensure_dirs():
    """Create every directory the app writes into. Cheap, idempotent, and it
    means no route ever has to guess whether its output folder exists."""
    for d in (CHARACTERS_DIR, DATASETS_DIR, DPO_DIR, LORAS_DIR, GGUF_DIR,
              MODELS_DIR, TRAINING_INPUT, VOICE_SAMPLES, AUDIO_DIR, IPC_DIR,
              REVIEW_DIR, GEN_REQUESTS, GEN_RESPONSES, EVAL_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


def rel(path):
    """Store paths relative to the project root wherever possible — an absolute
    Windows path baked into a character config breaks the moment the folder
    moves (audit finding 13)."""
    if not path:
        return path
    try:
        p = os.path.abspath(path)
        if os.path.commonpath([p, PROJECT_ROOT]) == PROJECT_ROOT:
            return os.path.relpath(p, PROJECT_ROOT).replace("\\", "/")
    except ValueError:
        pass          # different drive — nothing sensible to relativize against
    return path


def abspath(path):
    """Inverse of rel(): resolve a stored path back to something openable.

    rel() is meant to store paths relative to PROJECT_ROOT ("data/gguf_output/x.gguf"),
    but a few writers (the Characters-tab adapter/base-GGUF dropdowns, persona.py's
    CLI registration path) have stored paths relative to GGUF_DIR instead
    ("gguf_output/x.gguf", missing the "data/" segment) — those silently failed to
    resolve here, which showed up as "adapter ... missing, skipping" at engine start
    with no error surfaced anywhere else. Try the documented PROJECT_ROOT-relative
    form first; fall back to resolving against GGUF_DIR's parent (DATA_ROOT) for the
    older/inconsistent form before giving up.
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    primary = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(primary):
        return primary
    fallback = os.path.join(DATA_ROOT, path)
    return fallback if os.path.exists(fallback) else primary
