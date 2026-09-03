"""
core/generate.py — the one synthetic-data generator.

Replaces the two overlapping implementations that used to live in pipeline.py
(topic-driven, LM Studio) and persona_forge.py (profile-driven, local engine).
Both paths now share one endpoint layer, one JSON-salvage parser, one dedup
strategy, and one supervision hook.

Sources
    engine     — the running llama-server (the parent/base model)
    lmstudio   — any OpenAI-compatible endpoint, LM Studio by default
    claude     — writes a batch spec for Claude Code to answer on disk
    import     — a file the user drops in (handled in app.py, not here)

Supervision
    supervise_pct 0..100 decides what fraction of accepted samples are held in
    the review queue (core/dataset.py) instead of being written straight to the
    dataset file. The rest stream in as they are produced.
"""

import os
import json
import time
import random
import threading
import urllib.request

import config
import dataset as ds
import persona as pf

# ─────────────────────────────────────────────────────────────
#  LIVE STATE  (polled by the Data tab)
# ─────────────────────────────────────────────────────────────

gen_state = {
    "running": False,
    "status": "idle",          # idle | generating | waiting_claude | done | error | stopped
    "source": None,
    "name": None,
    "added": 0,                # written into the dataset
    "held": 0,                 # diverted to the review queue
    "rejected": 0,             # dropped as duplicates / malformed
    "target": 0,
    "batch": 0,
    "batches": 0,
    "error": None,
    "log": [],
    "started": None,
}

_stop = threading.Event()
_thread = None
LOG_CAP = 400


def _log(msg):
    print(f"[gen] {msg}")
    gen_state["log"].append(msg)
    if len(gen_state["log"]) > LOG_CAP:
        del gen_state["log"][:-LOG_CAP]


def _reset(name, source, target):
    _stop.clear()
    gen_state.update({
        "running": True, "status": "generating", "source": source, "name": name,
        "added": 0, "held": 0, "rejected": 0, "target": target,
        "batch": 0, "batches": 0, "error": None, "log": [], "started": time.time(),
    })


def stop():
    _stop.set()
    gen_state["status"] = "stopped"
    return True


def is_running():
    return bool(gen_state.get("running"))


# ─────────────────────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────────────────────

def probe(url, timeout=4):
    """Is there an OpenAI-compatible server on the other end, and what models
    does it have? Works for llama-server and LM Studio alike."""
    url = (url or "").rstrip("/")
    try:
        req = urllib.request.Request(url + "/v1/models")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        models = [m.get("id") for m in data.get("data", []) if m.get("id")]
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "models": [], "error": str(e)}


def _endpoint_for(source, lmstudio_url=None):
    if source == "lmstudio":
        return (lmstudio_url or config.LMSTUDIO_URL).rstrip("/")
    return config.ENGINE_URL.rstrip("/")


# ─────────────────────────────────────────────────────────────
#  PERSONA LOADING
# ─────────────────────────────────────────────────────────────

def load_persona(name):
    """Return (master_profile, persona_md). Either may be empty — generation
    still works from a plain persona string, it's just less varied."""
    pdir = os.path.join(config.TRAINING_INPUT, os.path.basename(str(name)))
    master, md = {}, ""
    mp = os.path.join(pdir, "master_persona.json")
    if os.path.exists(mp):
        try:
            with open(mp, encoding="utf-8") as f:
                master = json.load(f)
        except (OSError, json.JSONDecodeError):
            master = {}
    pm = os.path.join(pdir, "persona.md")
    if os.path.exists(pm):
        with open(pm, encoding="utf-8") as f:
            md = f.read()
    return master, md


def topic_pool(name):
    master, _ = load_persona(name)
    return pf._topic_pool(master) if master else []


# ─────────────────────────────────────────────────────────────
#  FORMATTING  (instruct vs thinking vs multi-turn)
# ─────────────────────────────────────────────────────────────

def _shape_record(ex, fmt):
    """Turn a generated {instruction, output, reasoning?} into the on-disk
    record for the requested training format."""
    inst = (ex.get("instruction") or "").strip()
    out = (ex.get("output") or "").strip()
    reasoning = (ex.get("reasoning") or "").strip()
    if not inst or not out:
        return None
    if fmt == "thinking":
        rec = {"instruction": inst, "input": "", "output": out}
        rec["reasoning"] = reasoning or "(no reasoning produced)"
        return rec
    if fmt == "chat":
        return {"messages": [{"role": "user", "content": inst},
                             {"role": "assistant", "content": out}]}
    return {"instruction": inst, "input": "", "output": out}


# ─────────────────────────────────────────────────────────────
#  THE RUN LOOP
# ─────────────────────────────────────────────────────────────

def start(cfg):
    """Kick off a generation run in a background thread. cfg keys:

        name, source, lmstudio_url, model, total, batch_size, temperature,
        max_tokens, timeout_s, format, dedupe_threshold, supervise_pct,
        topics (optional override list), fresh, persona_override
    """
    global _thread
    if is_running():
        raise RuntimeError("A generation run is already in progress.")
    name = os.path.basename(str(cfg.get("name") or "")).strip()
    if not name:
        raise ValueError("A dataset name is required.")

    _thread = threading.Thread(target=_run, args=(name, cfg), daemon=True)
    _thread.start()
    return True


def _run(name, cfg):
    source = cfg.get("source", "engine")
    total = int(cfg.get("total", config.GEN_TOTAL))
    batch_size = max(1, int(cfg.get("batch_size", config.GEN_BATCH)))
    temperature = float(cfg.get("temperature", config.GEN_TEMPERATURE))
    max_tokens = int(cfg.get("max_tokens", 2200))
    timeout_s = int(cfg.get("timeout_s", config.GEN_TIMEOUT_S))
    fmt = cfg.get("format", "instruct")
    threshold = float(cfg.get("dedupe_threshold", config.GEN_DEDUPE))
    supervise = max(0, min(100, int(cfg.get("supervise_pct", 0))))
    url = _endpoint_for(source, cfg.get("lmstudio_url"))
    model = cfg.get("model") or None

    _reset(name, source, total)
    rng = random.Random()

    try:
        if source == "claude":
            _run_claude_handshake(name, cfg)
            return

        if source == "book":
            import bookextract
            gen_state["target"] = 0   # chunk count, not a row target — set once known
            bookextract.run(
                dataset_name=name,
                persona_name=cfg.get("persona_name") or name,
                filename=cfg["book_file"],
                protagonist=cfg.get("protagonist") or "the protagonist",
                log=_log, stop_event=_stop, state=gen_state,
                source=cfg.get("book_source", "engine"),
                lmstudio_url=cfg.get("lmstudio_url"), model=model,
                temperature=temperature, timeout_s=timeout_s,
                max_tokens=int(cfg.get("max_tokens", 1800)),
                target_chars=int(cfg.get("chunk_chars", 3000)),
                max_chars=int(cfg.get("chunk_chars", 3000)) + 1500,
                dedupe_threshold=float(cfg.get("dedupe_threshold", 0.7)),
                supervise_pct=supervise,
                max_chunks=cfg.get("max_chunks"))
            return

        # The profile usually lives under the dataset's own name, but the UI
        # lets you point a dataset at a different persona folder.
        master, persona_md = load_persona(cfg.get("persona_name") or name)
        if cfg.get("persona_override"):
            persona_md = cfg["persona_override"]
        if not master and not persona_md:
            raise RuntimeError(
                f"No persona found for '{name}'. Build one in the Persona tab "
                f"first, or paste a persona summary into the override box.")

        # ── seed the dedup pool from what already exists ──
        seen_outputs, seen_instructions = [], []
        if not cfg.get("fresh"):
            rows, _ = ds.read_rows(name)
            seen_outputs = [r["output"] for r in rows if r["output"]]
            seen_instructions = [r["instruction"] for r in rows if r["instruction"]]
            if rows:
                _log(f"resuming — {len(rows)} existing examples seed the dedup pool")
        else:
            path = ds.dataset_path(name)
            if os.path.exists(path):
                ds._snapshot(path, "pre-fresh-gen")
                open(path, "w", encoding="utf-8").close()
                _log("fresh run — previous dataset snapshotted and cleared")
        seen_outputs.extend(master.get("sample_voice_lines", []))

        topics = cfg.get("topics") or None
        cycler = _Cycler(master, topics, seed=cfg.get("seed"))

        batches = max(1, -(-total // batch_size))
        gen_state["batches"] = batches
        _log(f"target {total} · batches of {batch_size} · temp {temperature} · "
             f"format {fmt} · supervise {supervise}% · via {source}")

        stalled = 0
        while gen_state["added"] + gen_state["held"] < total:
            if _stop.is_set():
                _log("stopped by user")
                gen_state["status"] = "stopped"
                break
            gen_state["batch"] += 1
            want = min(batch_size, total - gen_state["added"] - gen_state["held"])
            combos = cycler.draw(want)

            try:
                batch = _generate_batch(
                    master, persona_md, combos, temperature, max_tokens,
                    timeout_s, url, model, fmt,
                    avoid=seen_instructions[-24:] if seen_instructions else None)
            except Exception as e:
                _log(f"batch {gen_state['batch']} failed: {e}")
                stalled += 1
                if stalled >= 4:
                    raise RuntimeError(f"Four consecutive batches failed. Last error: {e}")
                continue

            kept = 0
            for ex in batch:
                out = (ex.get("output") or "").strip()
                inst = (ex.get("instruction") or "").strip()
                if not out or not inst:
                    gen_state["rejected"] += 1
                    continue
                if pf._is_near_duplicate(out, seen_outputs, threshold) or \
                   pf._is_near_duplicate(inst, seen_instructions, threshold):
                    gen_state["rejected"] += 1
                    continue
                rec = _shape_record(ex, fmt)
                if not rec:
                    gen_state["rejected"] += 1
                    continue
                seen_outputs.append(out)
                seen_instructions.append(inst)
                kept += 1

                # ── the supervision split ──
                if supervise and rng.randint(1, 100) <= supervise:
                    ds.queue_add(name, [rec], meta={
                        "source": source, "temperature": temperature,
                        "format": fmt, "batch": gen_state["batch"]})
                    gen_state["held"] += 1
                else:
                    ds.append_rows(name, [rec])
                    gen_state["added"] += 1

            _log(f"batch {gen_state['batch']}/{batches} — kept {kept}/{len(batch)} "
                 f"(in dataset {gen_state['added']}, held for review {gen_state['held']})")

            if kept == 0:
                stalled += 1
                # Not a bug: the generator hits a real diversity wall once the
                # profile's topic space is exhausted. Stop rather than burn
                # cycles producing rejects.
                if stalled >= 5:
                    _log("diversity wall — five batches produced nothing new. Stopping.")
                    break
            else:
                stalled = 0

        if gen_state["status"] != "stopped":
            gen_state["status"] = "done"
            _log(f"finished — {gen_state['added']} written, {gen_state['held']} awaiting "
                 f"review, {gen_state['rejected']} rejected as duplicates")

    except Exception as e:
        gen_state["status"] = "error"
        gen_state["error"] = str(e)
        _log(f"error: {e}")
    finally:
        gen_state["running"] = False


class _Cycler:
    """Forced-variety topic cycler. Uses the persona's own topic pool unless
    the user supplied an explicit topic list in the UI."""

    def __init__(self, master, topics=None, seed=None):
        pool = [t for t in (topics or []) if str(t).strip()] or pf._topic_pool(master or {})
        self._all = [(t, m) for t in pool for m in pf.MESSAGE_TYPES]
        self._rng = random.Random(seed)
        self._rng.shuffle(self._all)
        self._pool = list(self._all)

    def draw(self, n):
        out = []
        for _ in range(n):
            if not self._pool:
                self._pool = list(self._all)
                self._rng.shuffle(self._pool)
            out.append(self._pool.pop())
        return out


def _generate_batch(master, persona_md, combos, temperature, max_tokens,
                    timeout_s, url, model, fmt, avoid=None):
    """One request → a list of {instruction, output, reasoning?}."""
    avoid_block = ""
    if avoid:
        avoid_block = ("\nINSTRUCTIONS ALREADY USED (do not near-duplicate these):\n" +
                       "\n".join(f"- {t}" for t in avoid))
    assignments = "\n".join(
        f"{i+1}. topic: {t} | message type: {m}" for i, (t, m) in enumerate(combos))

    sys_msg = pf.GEN_SYSTEM
    if fmt == "thinking":
        sys_msg = sys_msg.replace(
            '[{"instruction":"a message someone sends them","input":"",'
            '"output":"their in-voice reply"}]',
            '[{"instruction":"a message someone sends them","input":"",'
            '"reasoning":"their brief private thinking before replying",'
            '"output":"their in-voice reply"}]'
        ).replace("no reasoning, no meta-commentary",
                  "with a short private reasoning span, then the reply")

    user = (
        f"PROFILE:\n{json.dumps(master, ensure_ascii=False)}\n\n"
        f"PERSONA SUMMARY:\n{persona_md}\n"
        f"{avoid_block}\n\n"
        f"ASSIGNMENTS ({len(combos)} examples, one per line):\n{assignments}\n\n"
        f"Generate exactly {len(combos)} examples, one per assignment above, "
        f"in order. Return ONLY the JSON array."
    )

    raw = pf._complete(
        [{"role": "system", "content": sys_msg},
         {"role": "user", "content": user}],
        temperature=temperature, max_tokens=max_tokens,
        timeout=timeout_s, url=url, model=model)

    arr = _salvage_json_array(raw)
    combo_topics = {str(t).lower().strip() for t, _ in combos}
    out = []
    for ex in arr:
        if not isinstance(ex, dict):
            continue
        inst = str(ex.get("instruction", "")).strip()
        outp = str(ex.get("output", "")).strip()
        if not inst or not outp:
            continue
        low = inst.lower().strip()
        # Reject the model leaking the internal scaffolding into the fake
        # instruction, or echoing the instruction back inside the answer.
        # Both are real failure modes, not hypothetical.
        if "message type" in low or " | " in inst or low in combo_topics:
            continue
        if len(inst) > 25 and low in outp.lower():
            continue
        if fmt != "thinking" and "<think" in outp.lower():
            continue
        out.append({"instruction": inst, "output": outp,
                    "reasoning": str(ex.get("reasoning", "")).strip()})
    return out


def _salvage_json_array(text):
    """One parser for every source. Models wrap arrays in fences and prose,
    trail commas, and occasionally truncate mid-array; recover what we can
    rather than throwing the whole batch away."""
    text = (text or "").strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                text = part
                break
    start, end = text.find("["), text.rfind("]")
    if start == -1:
        return []
    body = text[start:end + 1] if end > start else text[start:]

    import re as _re
    for attempt in (body, _re.sub(r",\s*([}\]])", r"\1", body), body + "]"):
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            continue

    # Last resort: pull complete {...} objects out one at a time. A truncated
    # array still yields every object that finished.
    out = []
    depth, buf, in_str, esc = 0, "", False, False
    for ch in body:
        if depth:
            buf += ch
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if not depth:
                buf = "{"
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    out.append(json.loads(buf))
                except json.JSONDecodeError:
                    pass
                buf = ""
    return out


# ─────────────────────────────────────────────────────────────
#  CLAUDE CODE HANDSHAKE
# ─────────────────────────────────────────────────────────────
# No API key, no network call from this app. We write a self-contained request
# file; you point Claude Code at it; it writes the answers back next door and
# this picks them up.

def _run_claude_handshake(name, cfg):
    master, persona_md = load_persona(cfg.get("persona_name") or name)
    total = int(cfg.get("total", 100))
    fmt = cfg.get("format", "instruct")
    topics = cfg.get("topics") or pf._topic_pool(master or {})

    os.makedirs(config.GEN_REQUESTS, exist_ok=True)
    os.makedirs(config.GEN_RESPONSES, exist_ok=True)
    req_path = os.path.join(config.GEN_REQUESTS, f"{name}.json")
    resp_path = os.path.join(config.GEN_RESPONSES, f"{name}.jsonl")

    spec = {
        "dataset": name,
        "requested": total,
        "format": fmt,
        "created": time.time(),
        "write_answers_to": resp_path,
        "instructions": (
            f"Generate {total} fine-tuning examples in this person's voice. "
            f"Write ONE JSON object per line (JSONL) to the path in "
            f"'write_answers_to'. Each line: "
            + ('{"instruction": "...", "input": "", "reasoning": "...", "output": "..."}'
               if fmt == "thinking"
               else '{"instruction": "...", "input": "", "output": "..."}')
            + ". Vary the topic and message type across the list — use the "
              "topics and message_types arrays below. Keep outputs short and "
              "in-voice; match the persona's typing style exactly. Do not copy "
              "sample_voice_lines verbatim; they are voice reference only."
        ),
        "persona_summary": persona_md,
        "profile": master,
        "topics": topics,
        "message_types": pf.MESSAGE_TYPES,
    }
    with open(req_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)

    gen_state["status"] = "waiting_claude"
    gen_state["running"] = False
    _log(f"wrote request spec → {config.rel(req_path)}")
    _log(f"run Claude Code against it, then click 'Import Claude output' "
         f"(expects {config.rel(resp_path)})")


def claude_request_path(name):
    return os.path.join(config.GEN_REQUESTS, f"{os.path.basename(str(name))}.json")


def collect_claude(name, supervise_pct=0, dedupe_threshold=None):
    """Import whatever Claude Code wrote back, applying the same dedup and
    supervision rules as a live generation run."""
    resp_path = os.path.join(config.GEN_RESPONSES, f"{os.path.basename(str(name))}.jsonl")
    if not os.path.exists(resp_path):
        raise FileNotFoundError(
            f"Nothing at {config.rel(resp_path)} yet — run Claude Code against "
            f"the request spec first.")

    threshold = float(dedupe_threshold if dedupe_threshold is not None else config.GEN_DEDUPE)
    existing, _ = ds.read_rows(name)
    seen = [r["output"] for r in existing if r["output"]]
    rng = random.Random()
    added = held = rejected = 0

    with open(resp_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                rejected += 1
                continue
            out = str(ex.get("output", "")).strip()
            if not out or pf._is_near_duplicate(out, seen, threshold):
                rejected += 1
                continue
            rec = _shape_record(ex, "thinking" if ex.get("reasoning") else "instruct")
            if not rec:
                rejected += 1
                continue
            seen.append(out)
            if supervise_pct and rng.randint(1, 100) <= int(supervise_pct):
                ds.queue_add(name, [rec], meta={"source": "claude"})
                held += 1
            else:
                ds.append_rows(name, [rec])
                added += 1

    # Move the consumed file aside so a second click can't double-import it.
    os.replace(resp_path, resp_path + f".imported-{int(time.time())}")
    return {"added": added, "held": held, "rejected": rejected}


# ─────────────────────────────────────────────────────────────
#  SINGLE-ROW REGENERATION  (the Data tab's per-row Regenerate button)
# ─────────────────────────────────────────────────────────────

def regenerate_one(name, instruction, source="engine", lmstudio_url=None,
                   model=None, temperature=0.9, max_tokens=400, timeout_s=90):
    """Produce a fresh answer for an existing prompt, keeping the prompt."""
    master, persona_md = load_persona(name)
    url = _endpoint_for(source, lmstudio_url)
    sys_msg = (
        "You reply as one specific person, in their voice, given their profile. "
        "Reply with ONLY the message they would send — no quotes, no name "
        "prefix, no preamble, no explanation. Keep it short: usually one "
        "sentence, rarely two. Match their typing style and punctuation habits "
        "exactly as described in the profile."
    )
    user = (f"PROFILE:\n{json.dumps(master, ensure_ascii=False)}\n\n"
            f"PERSONA SUMMARY:\n{persona_md}\n\n"
            f"Someone says to them:\n{instruction}\n\nTheir reply:")
    text = pf._complete([{"role": "system", "content": sys_msg},
                         {"role": "user", "content": user}],
                        temperature=temperature, max_tokens=max_tokens,
                        timeout=timeout_s, url=url, model=model)
    return text.strip().strip('"')
