"""
core/dataset.py — everything that happens to a dataset AFTER it is generated
and BEFORE it is trained on.

Read, edit, regenerate a single row, analyze for problems, and apply cleanup
operations. Also owns the supervision review queue: when generation runs with
supervise_pct > 0, a fraction of new samples land here for accept / edit /
regenerate / reject instead of going straight into the file.

Every write is atomic (temp + replace) and every destructive operation snapshots
the file to datasets/_snapshots/ first, so nothing here can lose work.
"""

import os
import re
import json
import time
import random
import shutil
import difflib

import config

SNAPSHOT_DIR = os.path.join(config.DATASETS_DIR, "_snapshots")

URL_RE   = re.compile(r"(https?://\S+)|(\bwww\.\S+)|"
                      r"(\b[a-zA-Z0-9-]+\.(?:com|net|org|io|gg|tv|co|me|xyz|app|dev|gov|edu|ai)\b\S*)",
                      re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
HANDLE_RE = re.compile(r"(?<![\w/])@[A-Za-z0-9_]{2,}")


# ─────────────────────────────────────────────────────────────
#  PATHS + IO
# ─────────────────────────────────────────────────────────────

def dataset_path(name):
    """Resolve a dataset name to its file. Name is sanitized to a bare stem so
    a crafted name can't write outside datasets/."""
    stem = os.path.basename(str(name or "")).replace(".jsonl", "").strip()
    if not stem:
        raise ValueError("Dataset name is required")
    return os.path.join(config.DATASETS_DIR, f"{stem}.jsonl")


def list_datasets():
    out = []
    if not os.path.isdir(config.DATASETS_DIR):
        return out
    for f in sorted(os.listdir(config.DATASETS_DIR)):
        if not f.endswith(".jsonl"):
            continue
        p = os.path.join(config.DATASETS_DIR, f)
        try:
            with open(p, encoding="utf-8") as fh:
                n = sum(1 for line in fh if line.strip())
        except OSError:
            n = 0
        out.append({
            "name": f[:-6],
            "rows": n,
            "bytes": os.path.getsize(p),
            "modified": os.path.getmtime(p),
        })
    return out


def read_rows(name):
    """Return [{idx, instruction, input, output, reasoning, _raw}] for every
    parseable line, plus a list of line numbers that failed to parse."""
    path = dataset_path(name)
    rows, bad = [], []
    if not os.path.exists(path):
        return rows, bad
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad.append(i + 1)
                continue
            rows.append(_normalize(obj, len(rows)))
    return rows, bad


def _normalize(obj, idx):
    """Flatten the several shapes a row can arrive in (alpaca-style, chat
    messages, plain prompt/response) into one editable shape."""
    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        instruction = output = reasoning = ""
        for m in obj["messages"]:
            role = m.get("role")
            if role == "user" and not instruction:
                instruction = m.get("content", "")
            elif role == "assistant" and not output:
                output = m.get("content", "")
        think = re.search(r"<think>(.*?)</think>", output, re.S)
        if think:
            reasoning = think.group(1).strip()
            output = re.sub(r"<think>.*?</think>", "", output, flags=re.S).strip()
        return {"idx": idx, "instruction": instruction, "input": "",
                "output": output, "reasoning": reasoning, "_raw": obj}

    instruction = (obj.get("instruction") or obj.get("prompt")
                   or obj.get("question") or obj.get("input") or "")
    output = (obj.get("output") or obj.get("answer") or obj.get("response")
              or obj.get("completion") or "")
    return {
        "idx": idx,
        "instruction": instruction,
        "input": obj.get("input", "") if obj.get("instruction") else "",
        "output": output,
        "reasoning": obj.get("reasoning", "") or obj.get("thinking", "") or "",
        "_raw": obj,
    }


def _to_record(row):
    """Serialize an edited row back to the on-disk shape."""
    rec = {"instruction": row.get("instruction", ""),
           "input": row.get("input", "") or "",
           "output": row.get("output", "")}
    if row.get("reasoning"):
        rec["reasoning"] = row["reasoning"]
    return rec


def _snapshot(path, tag):
    """Copy the current file aside before a destructive edit."""
    if not os.path.exists(path):
        return None
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = os.path.join(SNAPSHOT_DIR,
                        f"{os.path.basename(path)[:-6]}.{stamp}.{tag}.jsonl")
    shutil.copy2(path, dest)
    # keep the last 20 snapshots per dataset, not every one ever taken
    stem = os.path.basename(path)[:-6] + "."
    snaps = sorted(f for f in os.listdir(SNAPSHOT_DIR) if f.startswith(stem))
    for old in snaps[:-20]:
        try:
            os.remove(os.path.join(SNAPSHOT_DIR, old))
        except OSError:
            pass
    return dest


def write_rows(name, rows, tag="edit"):
    """Atomically rewrite the whole dataset from `rows`, snapshotting first."""
    path = dataset_path(name)
    snap = _snapshot(path, tag)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_to_record(r), ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return {"rows": len(rows), "snapshot": os.path.basename(snap) if snap else None}


def append_rows(name, records):
    """Append raw records without rewriting the file — the generation path."""
    path = dataset_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def list_snapshots(name):
    if not os.path.isdir(SNAPSHOT_DIR):
        return []
    stem = os.path.basename(str(name)) + "."
    out = []
    for f in sorted(os.listdir(SNAPSHOT_DIR), reverse=True):
        if f.startswith(stem):
            p = os.path.join(SNAPSHOT_DIR, f)
            out.append({"file": f, "bytes": os.path.getsize(p),
                        "modified": os.path.getmtime(p)})
    return out


def restore_snapshot(name, snapshot_file):
    src = os.path.join(SNAPSHOT_DIR, os.path.basename(snapshot_file))
    if not os.path.exists(src):
        raise FileNotFoundError(snapshot_file)
    path = dataset_path(name)
    _snapshot(path, "pre-restore")
    shutil.copy2(src, path)
    return True


# ─────────────────────────────────────────────────────────────
#  ANALYSIS
# ─────────────────────────────────────────────────────────────

def _similar(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _approx_tokens(text):
    """Rough token count without loading a tokenizer — good enough to flag
    rows that will be truncated at train time, and instant."""
    return int(len(text) / 3.6) + 1


def analyze(name, dupe_threshold=0.62, max_seq_length=1024,
            min_output_chars=8, persona_text=""):
    """Produce the cleanup report the Data tab renders. Returns per-issue lists
    of row indices plus a summary, without changing anything on disk."""
    rows, bad_lines = read_rows(name)
    n = len(rows)

    issues = {
        "duplicates": [],      # [{a, b, score}]
        "too_long": [],        # would be truncated
        "too_short": [],
        "empty": [],
        "has_url": [],
        "has_phone": [],
        "has_email": [],
        "has_handle": [],
        "drift": [],           # style far from persona.md
        "no_reasoning": [],
    }

    # ── near-duplicate scan, on outputs (the part that actually repeats) ──
    # O(n^2) is fine to a few thousand rows and avoids an embedding dependency;
    # a cheap length prefilter keeps the comparison count down.
    outs = [(i, r["output"].strip()) for i, r in enumerate(rows)]
    for a in range(len(outs)):
        ia, ta = outs[a]
        if not ta:
            continue
        for b in range(a + 1, len(outs)):
            ib, tb = outs[b]
            if not tb:
                continue
            if abs(len(ta) - len(tb)) > max(len(ta), len(tb)) * 0.5:
                continue
            score = _similar(ta, tb)
            if score >= dupe_threshold:
                issues["duplicates"].append({"a": ia, "b": ib, "score": round(score, 3)})

    # ── per-row checks ──
    style = style_profile_text(persona_text) if persona_text else None
    for i, r in enumerate(rows):
        instr, out = r["instruction"].strip(), r["output"].strip()
        if not instr or not out:
            issues["empty"].append(i)
            continue
        toks = _approx_tokens(instr + out + (r["reasoning"] or ""))
        if toks > max_seq_length:
            issues["too_long"].append({"idx": i, "tokens": toks})
        if len(out) < min_output_chars:
            issues["too_short"].append(i)
        blob = instr + " " + out
        if URL_RE.search(blob):
            issues["has_url"].append(i)
        if PHONE_RE.search(blob):
            issues["has_phone"].append(i)
        if EMAIL_RE.search(blob):
            issues["has_email"].append(i)
        if HANDLE_RE.search(blob):
            issues["has_handle"].append(i)
        if not r["reasoning"]:
            issues["no_reasoning"].append(i)
        if style:
            d = style_distance(out, style)
            if d > 0.45:
                issues["drift"].append({"idx": i, "distance": round(d, 3)})

    dupe_rows = sorted({d["b"] for d in issues["duplicates"]})
    return {
        "name": name,
        "total": n,
        "bad_lines": bad_lines,
        "issues": issues,
        "summary": {
            "duplicates": len(dupe_rows),
            "duplicate_pairs": len(issues["duplicates"]),
            "too_long": len(issues["too_long"]),
            "too_short": len(issues["too_short"]),
            "empty": len(issues["empty"]),
            "pii": len(set(issues["has_url"]) | set(issues["has_phone"])
                       | set(issues["has_email"])),
            "drift": len(issues["drift"]),
            "no_reasoning": len(issues["no_reasoning"]),
            "unparseable": len(bad_lines),
        },
        "clean_rows": max(0, n - len(dupe_rows) - len(issues["empty"])),
    }


def apply_clean(name, ops, dupe_threshold=0.62, max_seq_length=1024):
    """Apply cleanup operations and rewrite the file.

    ops is a list of any of: drop_duplicates, drop_empty, drop_too_short,
    drop_too_long, strip_urls, strip_phones, strip_emails, strip_handles,
    trim_whitespace.
    """
    rows, _ = read_rows(name)
    before = len(rows)
    ops = set(ops or [])
    removed = {"duplicates": 0, "empty": 0, "too_short": 0, "too_long": 0}
    scrubbed = 0

    if "trim_whitespace" in ops:
        for r in rows:
            r["instruction"] = re.sub(r"[ \t]+", " ", r["instruction"]).strip()
            r["output"] = r["output"].strip()

    for key, rx in (("strip_urls", URL_RE), ("strip_phones", PHONE_RE),
                    ("strip_emails", EMAIL_RE), ("strip_handles", HANDLE_RE)):
        if key in ops:
            for r in rows:
                new_i = rx.sub("", r["instruction"])
                new_o = rx.sub("", r["output"])
                if new_i != r["instruction"] or new_o != r["output"]:
                    scrubbed += 1
                r["instruction"] = re.sub(r"\s{2,}", " ", new_i).strip()
                r["output"] = re.sub(r"\s{2,}", " ", new_o).strip()

    if "drop_empty" in ops:
        keep = [r for r in rows if r["instruction"].strip() and r["output"].strip()]
        removed["empty"] = len(rows) - len(keep)
        rows = keep

    if "drop_too_short" in ops:
        keep = [r for r in rows if len(r["output"].strip()) >= 8]
        removed["too_short"] = len(rows) - len(keep)
        rows = keep

    if "drop_too_long" in ops:
        keep = [r for r in rows
                if _approx_tokens(r["instruction"] + r["output"] + (r["reasoning"] or ""))
                <= max_seq_length]
        removed["too_long"] = len(rows) - len(keep)
        rows = keep

    if "drop_duplicates" in ops:
        keep, seen = [], []
        for r in rows:
            out = r["output"].strip()
            if any(_similar(out, s) >= dupe_threshold for s in seen):
                removed["duplicates"] += 1
                continue
            seen.append(out)
            keep.append(r)
        rows = keep

    res = write_rows(name, rows, tag="clean")
    return {"before": before, "after": len(rows), "removed": removed,
            "scrubbed": scrubbed, **res}


# ─────────────────────────────────────────────────────────────
#  STYLE PROFILE  (shared by cleanup drift-flagging and the Evaluate tab)
# ─────────────────────────────────────────────────────────────

def style_profile_text(text):
    """Measure the handful of style dimensions that are actually observable in
    plain text — the ones bob's persona is literally specified in terms of
    ("lowercase, drops apostrophes, sparse punctuation")."""
    text = (text or "").strip()
    if not text:
        return None
    letters = [c for c in text if c.isalpha()]
    words = re.findall(r"[A-Za-z']+", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    contraction_slots = len(re.findall(
        r"\b(?:dont|cant|wont|im|ive|its|thats|youre|didnt|doesnt|isnt|hes|shes|theyre|"
        r"don't|can't|won't|i'm|i've|it's|that's|you're|didn't|doesn't|isn't|he's|she's|they're)\b",
        text, re.I))
    apostrophes = len(re.findall(r"\w'\w", text))
    return {
        "lowercase_ratio": (sum(1 for c in letters if c.islower()) / len(letters)) if letters else 0.0,
        "avg_word_len": (sum(len(w) for w in words) / len(words)) if words else 0.0,
        "avg_sentence_words": (len(words) / len(sentences)) if sentences else 0.0,
        "punct_density": sum(1 for c in text if c in ",;:—-()\"") / max(len(text), 1),
        "question_ratio": text.count("?") / max(len(sentences), 1),
        "exclaim_ratio": text.count("!") / max(len(sentences), 1),
        "apostrophe_drop": 1.0 - (apostrophes / contraction_slots) if contraction_slots else 0.0,
        "chars": len(text),
        "words": len(words),
        "sentences": len(sentences),
    }


_STYLE_DIMS = ("lowercase_ratio", "avg_word_len", "avg_sentence_words",
               "punct_density", "apostrophe_drop")
_STYLE_SCALE = {"lowercase_ratio": 1.0, "avg_word_len": 4.0,
                "avg_sentence_words": 20.0, "punct_density": 0.08,
                "apostrophe_drop": 1.0}


def style_distance(text, reference):
    """Normalized 0..1-ish distance between a sample's style and a reference
    profile. Used to flag drift, not to be precise."""
    prof = style_profile_text(text)
    if not prof or not reference:
        return 0.0
    diffs = []
    for k in _STYLE_DIMS:
        scale = _STYLE_SCALE[k] or 1.0
        diffs.append(min(1.0, abs(prof[k] - reference.get(k, 0)) / scale))
    return sum(diffs) / len(diffs)


def dataset_style_profile(name):
    """Aggregate style profile over every output in a dataset — the training
    data's own voice, which is what a trained adapter should end up matching."""
    rows, _ = read_rows(name)
    blob = "\n".join(r["output"] for r in rows if r["output"].strip())
    prof = style_profile_text(blob)
    if prof:
        prof["source_rows"] = len(rows)
    return prof


# ─────────────────────────────────────────────────────────────
#  SUPERVISION REVIEW QUEUE
# ─────────────────────────────────────────────────────────────
# When generation runs with supervise_pct > 0, that fraction of new samples is
# diverted here instead of being appended straight to the dataset. The Data tab
# drains this queue with accept / edit / regenerate / reject.

def _queue_path(name):
    os.makedirs(config.REVIEW_DIR, exist_ok=True)
    return os.path.join(config.REVIEW_DIR, f"{os.path.basename(str(name))}.json")


def load_queue(name):
    p = _queue_path(name)
    if not os.path.exists(p):
        return []
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def _save_queue(name, items):
    p = _queue_path(name)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def queue_add(name, records, meta=None):
    """Hold generated records for human review."""
    items = load_queue(name)
    for rec in records:
        items.append({
            "id": f"r{int(time.time()*1000)}{random.randint(100,999)}",
            "record": rec,
            "meta": meta or {},
            "ts": time.time(),
            "status": "pending",
        })
    _save_queue(name, items)
    return len(items)


def queue_stats(name):
    items = load_queue(name)
    pending = [i for i in items if i["status"] == "pending"]
    return {
        "pending": len(pending),
        "held_total": len(items),
        "accepted": sum(1 for i in items if i["status"] == "accepted"),
        "rejected": sum(1 for i in items if i["status"] == "rejected"),
    }


def queue_decide(name, item_id, action, record=None):
    """action: accept | reject | edit (accept with replacement record).
    Accepted records are appended to the dataset immediately."""
    items = load_queue(name)
    for it in items:
        if it["id"] != item_id:
            continue
        if action in ("accept", "edit"):
            rec = record or it["record"]
            append_rows(name, [rec])
            it["record"] = rec
            it["status"] = "accepted"
        elif action == "reject":
            it["status"] = "rejected"
        else:
            raise ValueError(f"Unknown review action: {action}")
        it["decided"] = time.time()
        _save_queue(name, items)
        return it
    raise KeyError(item_id)


def queue_clear_decided(name):
    items = [i for i in load_queue(name) if i["status"] == "pending"]
    _save_queue(name, items)
    return len(items)
