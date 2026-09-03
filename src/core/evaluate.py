"""
core/evaluate.py — does the adapter actually work?

Four things the project had no answer for before:

  compare       one prompt, several adapters (and the bare base model) side by
                side, so "did DPO help or hurt" is answerable
  style_report  measure the outputs against the persona's own measurable style
                traits — lowercase ratio, apostrophe drops, sentence length
  memorization  flag generations that are near-verbatim training rows, the real
                risk when training 3 epochs on a few hundred examples
  regression    a pinned prompt set, re-run after each training pass and diffed
                against the previous run

Held-out loss is not here: it comes from the training run itself, via the
eval_split_pct config and the loss_history series in core/pipeline.py.
"""

import os
import json
import time
import difflib

import config
import dataset as ds
import store
from engine import MANAGER

REGRESSION_FILE = os.path.join(config.EVAL_DIR, "regression_prompts.json")
RUNS_DIR = os.path.join(config.EVAL_DIR, "runs")

DEFAULT_REGRESSION_PROMPTS = [
    "hey what are you up to",
    "whats your take on people using ai to write all their code",
    "i think we should just rewrite the whole thing from scratch",
    "explain how lora fine tuning works",
    "my build keeps failing and i have no idea why",
    "tell me something that happened to you recently",
    "whats the best thing you've built",
    "do you think local models will ever beat the big hosted ones",
]


# ─────────────────────────────────────────────────────────────
#  GENERATION HELPERS
# ─────────────────────────────────────────────────────────────

def _gen(prompt, char_id, persona="", temperature=0.8, max_tokens=200,
         enable_thinking=False):
    """One generation through the engine with a specific adapter active.
    char_id=None means the bare base model — that's the comparison baseline."""
    messages = []
    if persona:
        messages.append({"role": "system", "content": persona})
    messages.append({"role": "user", "content": prompt})
    return MANAGER.complete(messages, char_id=char_id, temperature=temperature,
                            max_tokens=max_tokens, enable_thinking=enable_thinking)


def compare(prompt, char_ids, temperature=0.8, max_tokens=200,
            include_base=True, use_persona=True):
    """Same prompt through several adapters. Returns one entry per variant.

    An adapter that isn't loaded into the running engine is reported as such
    rather than silently falling through to base-model output — that
    distinction is the entire point of this view.
    """
    if not MANAGER.is_ready():
        raise RuntimeError("Engine not running — start it from the Engine tab first.")

    variants = []
    if include_base:
        variants.append({"label": "base model", "char_id": None, "persona": ""})
    for cid in char_ids or []:
        char = store.get_character(cid)
        if not char:
            continue
        variants.append({
            "label": char.get("name", cid),
            "char_id": cid,
            "persona": char.get("persona", "") if use_persona else "",
        })

    loaded = {a["name"] for a in getattr(MANAGER, "adapters", [])}
    results = []
    for v in variants:
        entry = {"label": v["label"], "char_id": v["char_id"],
                 "adapter_loaded": v["char_id"] is None or v["char_id"] in loaded}
        t0 = time.time()
        try:
            if v["char_id"] and not entry["adapter_loaded"]:
                entry["text"] = ""
                entry["error"] = ("This character's adapter is not loaded in the "
                                  "running engine — restart the engine to include it.")
            else:
                entry["text"] = _gen(prompt, v["char_id"], v["persona"],
                                     temperature, max_tokens)
        except Exception as e:
            entry["text"] = ""
            entry["error"] = str(e)
        entry["seconds"] = round(time.time() - t0, 2)
        entry["style"] = ds.style_profile_text(entry.get("text", ""))
        results.append(entry)

    return {"prompt": prompt, "temperature": temperature, "results": results}


# ─────────────────────────────────────────────────────────────
#  STYLE SCORECARD
# ─────────────────────────────────────────────────────────────

STYLE_LABELS = {
    "lowercase_ratio":    "Lowercase ratio",
    "avg_word_len":       "Avg word length",
    "avg_sentence_words": "Words per sentence",
    "punct_density":      "Punctuation density",
    "apostrophe_drop":    "Apostrophe drop rate",
    "question_ratio":     "Questions per sentence",
    "exclaim_ratio":      "Exclamations per sentence",
}


def style_report(char_id, dataset_name, prompts=None, temperature=0.8,
                 max_tokens=200, n=None):
    """Generate against a set of prompts, then compare the output's measurable
    style against the training data's own style. A trained adapter should land
    close to its dataset; a big gap means the voice didn't transfer."""
    if not MANAGER.is_ready():
        raise RuntimeError("Engine not running — start it from the Engine tab first.")

    char = store.get_character(char_id)
    if not char:
        raise KeyError(f"No character {char_id}")

    reference = ds.dataset_style_profile(dataset_name)
    if not reference:
        raise ValueError(f"Dataset '{dataset_name}' is empty — nothing to compare against.")

    prompts = prompts or load_regression_prompts()
    if n:
        prompts = prompts[:int(n)]

    samples = []
    for p in prompts:
        try:
            text = _gen(p, char_id, char.get("persona", ""), temperature, max_tokens)
        except Exception as e:
            samples.append({"prompt": p, "text": "", "error": str(e)})
            continue
        samples.append({"prompt": p, "text": text,
                        "distance": round(ds.style_distance(text, reference), 3)})

    blob = "\n".join(s.get("text", "") for s in samples if s.get("text"))
    observed = ds.style_profile_text(blob) or {}

    dims = []
    for key, label in STYLE_LABELS.items():
        ref_v = reference.get(key)
        obs_v = observed.get(key)
        if ref_v is None or obs_v is None:
            continue
        scale = ds._STYLE_SCALE.get(key) or max(abs(ref_v), 1e-6)
        gap = min(1.0, abs(obs_v - ref_v) / scale)
        dims.append({
            "key": key, "label": label,
            "dataset": round(ref_v, 4), "model": round(obs_v, 4),
            "gap": round(gap, 3),
            "verdict": "close" if gap < 0.2 else ("drifting" if gap < 0.45 else "off"),
        })

    scored = [s["distance"] for s in samples if "distance" in s]
    return {
        "character": char.get("name", char_id),
        "char_id": char_id,
        "dataset": dataset_name,
        "overall_distance": round(sum(scored) / len(scored), 3) if scored else None,
        "dimensions": dims,
        "samples": samples,
    }


# ─────────────────────────────────────────────────────────────
#  MEMORIZATION
# ─────────────────────────────────────────────────────────────

def memorization_check(char_id, dataset_name, prompts=None, threshold=0.82,
                       temperature=0.8, max_tokens=200, n=12):
    """Is the adapter reciting its training data rather than generalising?

    Generates fresh answers, then finds the closest training output to each.
    Anything above the threshold is effectively memorised — expected to some
    degree, alarming if it's most of them.
    """
    if not MANAGER.is_ready():
        raise RuntimeError("Engine not running — start it from the Engine tab first.")

    char = store.get_character(char_id)
    if not char:
        raise KeyError(f"No character {char_id}")

    rows, _ = ds.read_rows(dataset_name)
    train_outputs = [r["output"].strip() for r in rows if r["output"].strip()]
    if not train_outputs:
        raise ValueError(f"Dataset '{dataset_name}' has no outputs to compare against.")

    # Prefer prompts the model was actually trained on — that's where
    # memorisation shows up, not on unseen prompts.
    prompts = prompts or [r["instruction"] for r in rows if r["instruction"].strip()][:int(n)]

    findings = []
    for p in prompts[:int(n)]:
        try:
            text = _gen(p, char_id, char.get("persona", ""), temperature, max_tokens).strip()
        except Exception as e:
            findings.append({"prompt": p, "error": str(e)})
            continue
        best, best_score = "", 0.0
        for t in train_outputs:
            score = difflib.SequenceMatcher(None, text.lower(), t.lower()).ratio()
            if score > best_score:
                best, best_score = t, score
        findings.append({
            "prompt": p, "generated": text,
            "closest_training_row": best,
            "similarity": round(best_score, 3),
            "memorized": best_score >= threshold,
        })

    scored = [f for f in findings if "similarity" in f]
    n_mem = sum(1 for f in scored if f["memorized"])
    return {
        "character": char.get("name", char_id),
        "dataset": dataset_name,
        "threshold": threshold,
        "checked": len(scored),
        "memorized": n_mem,
        "memorized_pct": round(100 * n_mem / len(scored), 1) if scored else 0,
        "mean_similarity": round(sum(f["similarity"] for f in scored) / len(scored), 3) if scored else 0,
        "findings": sorted(scored, key=lambda f: -f["similarity"]),
    }


# ─────────────────────────────────────────────────────────────
#  REGRESSION SET
# ─────────────────────────────────────────────────────────────

def load_regression_prompts():
    if os.path.exists(REGRESSION_FILE):
        try:
            with open(REGRESSION_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return list(DEFAULT_REGRESSION_PROMPTS)


def save_regression_prompts(prompts):
    os.makedirs(config.EVAL_DIR, exist_ok=True)
    clean = [str(p).strip() for p in prompts if str(p).strip()]
    tmp = REGRESSION_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    os.replace(tmp, REGRESSION_FILE)
    return clean


def _runs_path(char_id):
    os.makedirs(RUNS_DIR, exist_ok=True)
    return os.path.join(RUNS_DIR, f"{char_id}.json")


def list_runs(char_id):
    p = _runs_path(char_id)
    if not os.path.exists(p):
        return []
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def run_regression(char_id, temperature=0.8, max_tokens=200, label=None):
    """Run every pinned prompt and store the result, diffed against last time."""
    if not MANAGER.is_ready():
        raise RuntimeError("Engine not running — start it from the Engine tab first.")

    char = store.get_character(char_id)
    if not char:
        raise KeyError(f"No character {char_id}")

    prompts = load_regression_prompts()
    previous = list_runs(char_id)
    prev_map = {}
    if previous:
        prev_map = {r["prompt"]: r["text"] for r in previous[-1].get("results", [])}

    results = []
    for p in prompts:
        try:
            text = _gen(p, char_id, char.get("persona", ""), temperature, max_tokens)
        except Exception as e:
            results.append({"prompt": p, "text": "", "error": str(e)})
            continue
        prev = prev_map.get(p)
        results.append({
            "prompt": p,
            "text": text,
            "previous": prev,
            "changed": (prev is not None and prev.strip() != text.strip()),
            "drift": (round(1 - difflib.SequenceMatcher(None, prev.lower(), text.lower()).ratio(), 3)
                      if prev else None),
        })

    run = {
        "ts": time.time(),
        "label": label or time.strftime("%Y-%m-%d %H:%M"),
        "char_id": char_id,
        "character": char.get("name", char_id),
        "adapter": char.get("adapter_gguf", ""),
        "temperature": temperature,
        "results": results,
    }
    runs = previous + [run]
    runs = runs[-25:]          # keep the last 25 runs, not every one ever
    p = _runs_path(char_id)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)
    return run


# ─────────────────────────────────────────────────────────────
#  TRAINING-RUN HISTORY
# ─────────────────────────────────────────────────────────────

def save_training_run(lora_name, status_snapshot):
    """Archive a finished run's loss history so the Evaluate tab can show
    curves for past runs, not only the one still in memory."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    p = os.path.join(RUNS_DIR, f"_train_{os.path.basename(str(lora_name))}.json")
    history = []
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                history = json.load(f)
        except (OSError, json.JSONDecodeError):
            history = []
    history.append({
        "ts": time.time(),
        "lora_name": lora_name,
        "final_loss": status_snapshot.get("loss"),
        "final_eval_loss": status_snapshot.get("eval_loss"),
        "steps": status_snapshot.get("total_steps"),
        "epochs": status_snapshot.get("epoch"),
        "loss_history": status_snapshot.get("loss_history") or [],
    })
    history = history[-15:]
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False)
    os.replace(tmp, p)
    return len(history)


def training_runs(lora_name):
    p = os.path.join(RUNS_DIR, f"_train_{os.path.basename(str(lora_name))}.json")
    if not os.path.exists(p):
        return []
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
