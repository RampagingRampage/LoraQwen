"""
persona_forge.py — build a persona LoRA from raw documents (social posts,
journals, whatever you drop in training_input/<persona>/), using the local
Qwen3-8B engine itself to do the extraction. No thinking mode, no external
API calls — everything stays on this machine.

Pipeline:
    1. ingest      training_input/<persona>/*  →  chunks of a few paragraphs
    2. distill     each chunk updates a running trait-schema JSON
                    (master_persona.json) — the model reads its own running
                    notes + the new chunk and rewrites the schema
    3. write-up    master schema → a short persona string (for config.json's
                    dynamic system prompt) + a voice profile summary
    4. gendata     master schema → datasets/<persona>.jsonl, synthetic
                    instruction/response pairs in-voice, NO <think> tags
    5. train       (separate step — see train_persona() / CLI `train`)
                    hands the dataset to pipeline.run_training() +
                    run_gguf_conversion(), then registers the character.

Usage (always from the project root, not from inside tools/):
    python tools/persona_forge.py build   <persona_name>        # steps 1-3
    python tools/persona_forge.py gendata <persona_name> [-n 300]  # step 4
    python tools/persona_forge.py train   <persona_name>        # step 5
    python tools/persona_forge.py all     <persona_name> [-n 300]  # 1-4, prints
                                                            # the train command
"""

import os, sys, json, re, glob, argparse, textwrap, urllib.request, urllib.error, time, difflib, random

# Windows' console defaults to cp1252, which can't print this file's arrows/
# checkmarks -- force UTF-8 stdio instead of requiring PYTHONIOENCODING=utf-8
# to be set by hand every time.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# This script lives in tools/ but needs both its sibling tools (pipeline) and
# the root-level app modules (store, voice) importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # core/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # project root

import config as _cfg

# Endpoints and roots come from core/config.py (i.e. .env) rather than being
# hardcoded here, so the same generator can be pointed at llama-server, at
# LM Studio, or at any other OpenAI-compatible endpoint without editing code.
ENGINE_URL   = _cfg.ENGINE_URL
INPUT_ROOT   = _cfg.TRAINING_INPUT
DATASET_ROOT = _cfg.DATASETS_DIR

# ─────────────────────────────────────────────────────────────
#  TRAIT SCHEMA
# ─────────────────────────────────────────────────────────────
# Fixed shape the model fills in / refines chunk by chunk. Numeric dials are
# 0-10. Extend this freely — every new key just becomes one more thing the
# distiller is asked to track.

TRAIT_SCHEMA = {
    "voice_style": {
        "typing_style": "",       # capitalization, punctuation habits, emoji/abbreviation use
        "vocabulary": "",         # word choice, slang, jargon, favorite words/phrases
        "sentence_length": "",    # short/punchy vs long/rambling, fragment use
        "humor_style": "",        # sarcasm, deadpan, self-deprecating, puns, absent
        "formality": 0,           # 0 = very casual .. 10 = formal
    },
    "temperament": {
        "neurotic_level": 0,          # 0 = calm/stable .. 10 = anxious/reactive
        "manic_level": 0,             # 0 = low-energy/measured .. 10 = high-energy/impulsive
        "openness": 0,                # 0 = guarded .. 10 = overshares
        "agreeableness": 0,           # 0 = confrontational .. 10 = accommodating
        "emotional_volatility": 0,    # 0 = even-keeled .. 10 = swings hard
        "optimism": 0,                # 0 = cynical/bleak .. 10 = relentlessly upbeat
    },
    "values_worldview": {
        "core_beliefs": [],            # short phrases
        "recurring_opinions": [],
        "cares_about": [],
        "dismisses_or_mocks": [],
    },
    "content_interests": {
        "topics": [],                  # what they actually post about
        "recurring_references": [],    # in-jokes, shows, games, people, places
    },
    "social_patterns": {
        "how_they_argue": "",
        "how_they_comfort_others": "",
        "how_they_joke_with_friends": "",
        "public_vs_private": 0,        # 0 = very private .. 10 = performs for an audience
    },
    "quirks": {
        "verbal_tics": [],             # recurring words/phrases ("honestly", "ngl", excessive ellipses…)
        "topics_avoided": [],
        "contradictions": [],          # says X but does Y
    },
    "sample_voice_lines": [],          # verbatim standout quotes pulled straight from the source docs
}


# ─────────────────────────────────────────────────────────────
#  ENGINE CALL (talks directly to the running llama-server; independent of
#  llama_backend.MANAGER's process-tracking, so it works whether the engine
#  was started by this script, server.py, or is just already running)
# ─────────────────────────────────────────────────────────────

def _complete(messages, temperature=0.4, max_tokens=1200, timeout=120,
              url=None, model=None):
    payload = {
        "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if model:
        payload["model"] = model
    req = urllib.request.Request(
        (url or ENGINE_URL).rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        result = json.loads(r.read())
    return result["choices"][0]["message"].get("content", "") or ""


def _engine_alive(url=None):
    try:
        with urllib.request.urlopen((url or ENGINE_URL).rstrip("/") + "/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _atomic_write_json(path, data):
    """Write via a temp file + rename so a kill/crash mid-write can never
    leave a truncated/corrupted file — the rename is atomic, so readers only
    ever see the old complete file or the new complete file, never a partial
    one."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────
#  ROBUST JSON EXTRACTION  (models wrap JSON in prose/fences sometimes)
# ─────────────────────────────────────────────────────────────

def _extract_json_object(text):
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                text = p
                break
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    block = text[start:end + 1]
    try:
        return json.loads(block)
    except Exception:
        pass
    # light repair: trailing commas, stray control chars
    repaired = re.sub(r",\s*([}\]])", r"\1", block)
    repaired = "".join(ch for ch in repaired if ch >= " " or ch in "\n\t\r")
    try:
        return json.loads(repaired)
    except Exception:
        return None


def _deep_merge_schema(base, update):
    """Merge `update` into `base` following TRAIT_SCHEMA's shape: dicts merge
    key-by-key, lists get new unique items appended (capped), scalars/numbers
    get replaced only when the update actually provides a value."""
    if not isinstance(update, dict):
        return base
    for k, v in update.items():
        if k not in base:
            continue
        if isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge_schema(base[k], v)
        elif isinstance(base[k], list):
            if isinstance(v, list):
                for item in v:
                    if item and item not in base[k]:
                        base[k].append(item)
                base[k] = base[k][:40]
        elif isinstance(base[k], (int, float)):
            if isinstance(v, (int, float)):
                base[k] = max(0, min(10, v))
        else:
            if isinstance(v, str) and v.strip():
                base[k] = v.strip()
    return base


# ─────────────────────────────────────────────────────────────
#  1. INGEST + CHUNK
# ─────────────────────────────────────────────────────────────

def _load_documents(persona_dir):
    texts = []
    for path in sorted(glob.glob(os.path.join(persona_dir, "*"))):
        if os.path.isdir(path):
            continue
        name = os.path.basename(path)
        if name in ("master_persona.json", "persona.md"):
            continue
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".txt", ".md"):
                texts.append(open(path, encoding="utf-8", errors="ignore").read())
            elif ext == ".json":
                data = json.load(open(path, encoding="utf-8", errors="ignore"))
                texts.append(_flatten_json_text(data))
            elif ext in (".csv", ".tsv"):
                texts.append(open(path, encoding="utf-8", errors="ignore").read())
            elif ext == ".jsonl":
                lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
                for line in lines:
                    try:
                        texts.append(_flatten_json_text(json.loads(line)))
                    except Exception:
                        pass
        except Exception as e:
            print(f"  ⚠ could not read {name}: {e}")
    return texts


def _flatten_json_text(obj, depth=0):
    """Best-effort: pull human-readable strings out of an arbitrary JSON
    export (e.g. a social media data dump) without needing to know its shape."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            if k.lower() in ("text", "content", "body", "message", "caption", "post"):
                parts.append(_flatten_json_text(v, depth + 1))
        if parts:
            return "\n".join(p for p in parts if p)
        return "\n".join(_flatten_json_text(v, depth + 1) for v in obj.values() if depth < 3)
    if isinstance(obj, list):
        return "\n".join(_flatten_json_text(v, depth + 1) for v in obj if depth < 3)
    return ""


def _chunk_text(full_text, target_chars=1400):
    """Group text into ~target_chars chunks. Works on LINES as the atomic
    unit (not paragraphs) — handles both normal prose (blank-line-separated
    paragraphs collapse fine when grouped by line) and one-message-per-line
    exports (chat logs, message dumps) that have no blank lines at all, which
    would otherwise collapse into a single giant chunk."""
    lines = []
    for line in full_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Common export artifact: a trailing comma from a stripped list/CSV dump.
        if line.endswith(",") and not line.endswith(",\""):
            line = line[:-1].rstrip()
        if line:
            lines.append(line)

    chunks, buf = [], ""
    for ln in lines:
        if buf and len(buf) + len(ln) + 1 > target_chars:
            chunks.append(buf.strip())
            buf = ln
        else:
            buf = (buf + "\n" + ln) if buf else ln
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def gather_chunks(persona_name):
    persona_dir = os.path.join(INPUT_ROOT, persona_name)
    if not os.path.isdir(persona_dir):
        raise SystemExit(
            f"No folder at {persona_dir}/. Create it and drop documents in "
            f"(.txt, .md, .json, .jsonl, .csv) before running 'build'."
        )
    docs = _load_documents(persona_dir)
    if not docs:
        raise SystemExit(f"No readable documents found in {persona_dir}/.")
    chunks = []
    for doc in docs:
        chunks.extend(_chunk_text(doc))
    chunks = [c for c in chunks if len(c) > 40]  # drop near-empty scraps
    return chunks


# ─────────────────────────────────────────────────────────────
#  2. DISTILL — chunk-by-chunk running update of the trait schema
# ─────────────────────────────────────────────────────────────

DISTILL_SYSTEM = (
    "You extract a writing/personality voice-profile from real text samples, "
    "for building a fine-tuning persona, incrementally across many small "
    "batches of source text. You are shown the CURRENT profile (for context "
    "only, so you don't repeat what's already captured) and a NEW batch of "
    "source text. Output ONLY a PATCH: a small JSON object containing just "
    "the fields that should change, in the same nested shape as the profile. "
    "Rules:\n"
    "  - For list fields (arrays), include ONLY new items not already present "
    "in the current profile — never repeat existing items, never re-send the "
    "whole list.\n"
    "  - For numeric dials (0-10), include a field only if this new text "
    "shifts your estimate; omit fields you have no new evidence for.\n"
    "  - For text fields (typing_style, vocabulary, etc.), include only if "
    "this batch adds or refines something not already captured.\n"
    "  - Add at most 1-2 new verbatim standout quotes to sample_voice_lines, "
    "only if genuinely distinctive.\n"
    "  - Omit any top-level section entirely if this batch adds nothing to it.\n"
    "Output ONLY the patch JSON object — no commentary, no markdown fences, "
    "and never re-emit fields/items that are already in the current profile."
)


def distill_step(master, chunk_text):
    user = (
        f"CURRENT PROFILE (for context — do not repeat any of this back):\n"
        f"{json.dumps(master, ensure_ascii=False)}\n\n"
        f"NEW SOURCE TEXT:\n{chunk_text}\n\n"
        f"Return ONLY the patch (new/changed fields, same nested shape)."
    )
    raw = _complete(
        [{"role": "system", "content": DISTILL_SYSTEM},
         {"role": "user", "content": user}],
        temperature=0.3, max_tokens=900,
    )
    updated = _extract_json_object(raw)
    if updated is None:
        return master, False
    return _deep_merge_schema(master, updated), True


def build_master_persona(persona_name, limit=None, sample=None):
    if not _engine_alive():
        raise SystemExit(f"Engine not reachable at {ENGINE_URL} — start it first.")

    chunks = gather_chunks(persona_name)
    total_found = len(chunks)
    if sample and total_found > sample:
        # Random sample spread across the whole corpus, not just the first N —
        # matters for a large chronological export where the front and back
        # of the file can look very different.
        idx = sorted(random.sample(range(total_found), sample))
        chunks = [chunks[i] for i in idx]
        print(f"  ℹ sampling {sample} of {total_found} chunks (random, spread across the corpus)")
    elif limit:
        chunks = chunks[:limit]
    print(f"▶ {len(chunks)} chunks to distill for '{persona_name}'")

    persona_dir = os.path.join(INPUT_ROOT, persona_name)
    master_path = os.path.join(persona_dir, "master_persona.json")
    master = json.loads(json.dumps(TRAIT_SCHEMA))  # deep copy default

    if os.path.exists(master_path):
        try:
            master = _deep_merge_schema(master, json.load(open(master_path, encoding="utf-8")))
            print("  ↺ resuming from existing master_persona.json")
        except Exception:
            pass

    for i, chunk in enumerate(chunks, 1):
        t0 = time.time()
        master, ok = distill_step(master, chunk)
        _atomic_write_json(master_path, master)
        status = "ok" if ok else "parse-fail (kept prior state)"
        print(f"  [{i}/{len(chunks)}] {status} · {time.time()-t0:.1f}s")

    print(f"✓ master_persona.json written → {master_path}")

    persona_text = write_persona_string(master)
    persona_md_path = os.path.join(persona_dir, "persona.md")
    with open(persona_md_path, "w", encoding="utf-8") as f:
        f.write(persona_text)
    print(f"✓ persona.md written → {persona_md_path}")
    return master, persona_text


# ─────────────────────────────────────────────────────────────
#  3. WRITE-UP — schema → short system-prompt persona string
# ─────────────────────────────────────────────────────────────

WRITEUP_SYSTEM = (
    "You write a concise second-person persona description for a fine-tuned "
    "chatbot, based on a structured voice-profile JSON. Write 4-8 sentences "
    "starting with 'You are...'. Cover: how they write (style/vocabulary/"
    "punctuation), their temperament, what they care about and what they "
    "mock or avoid, and 1-2 verbatim phrases they'd actually say. No headers, "
    "no bullet points, plain prose only."
)


def write_persona_string(master):
    raw = _complete(
        [{"role": "system", "content": WRITEUP_SYSTEM},
         {"role": "user", "content": json.dumps(master, ensure_ascii=False)}],
        temperature=0.5, max_tokens=400,
    )
    return raw.strip()


# ─────────────────────────────────────────────────────────────
#  4. GENDATA — schema → synthetic non-thinking training set
# ─────────────────────────────────────────────────────────────

GEN_SYSTEM = (
    "You generate fine-tuning examples that teach a model to write in one "
    "specific person's voice, given their profile. Each example is a short "
    "user message and an in-voice reply — no reasoning, no meta-commentary, "
    "just the reply as that person would actually type it (respecting their "
    "typing style, vocabulary, humor, temperament, and opinions from the "
    "profile).\n\n"
    "You will be given a NUMBERED LIST OF (topic, message-type) ASSIGNMENTS — "
    "one per example, in order. Generate exactly one example per assignment, "
    "using that specific topic and message type. Do not substitute your own "
    "topic choice; the forced variety is intentional.\n\n"
    "CRITICAL — sample_voice_lines in the profile are voice REFERENCE ONLY, "
    "not content to output. Never copy one verbatim (or near-verbatim) as an "
    "'output'. Every output must be a NEW thing said in response to its own "
    "NEW instruction — a fresh scenario, not a repeat of an existing quote "
    "wearing a different question.\n\n"
    "CRITICAL — keep every 'output' SHORT: usually one sentence, rarely more "
    "than two, never a paragraph. Minimal punctuation: mostly skip commas and "
    "question marks, periods optional. This person is not an expert holding "
    "forth — they're a casual dabbler who types fast and moves on. If the "
    "profile's own sample lines are short and low-punctuation, match that, "
    "not the length of the instruction you were given.\n\n"
    "Output ONLY a JSON array, same order as the assignments: "
    '[{"instruction":"a message someone sends them","input":"",'
    '"output":"their in-voice reply"}]. No markdown fences, no prose before '
    "or after the array."
)

MESSAGE_TYPES = [
    "a direct task command (do this thing, fix that)",
    "casual check-in / small talk",
    "an opinion or hot-take question",
    "a technical problem they need help debugging",
    "a friend venting about something unrelated to them",
    "a disagreement or correction to push back on",
    "a joke or light banter exchange",
    "a 'what if' hypothetical worth chasing for its own sake",
    "reacting to a short status update / piece of news",
    "someone asking for their honest take on an idea",
]


def _topic_pool(master):
    pool = []
    ci = master.get("content_interests", {})
    pool.extend(ci.get("topics", []))
    pool.extend(ci.get("recurring_references", []))
    vw = master.get("values_worldview", {})
    pool.extend(vw.get("core_beliefs", []))
    pool.extend(vw.get("recurring_opinions", []))
    pool.extend(vw.get("cares_about", []))
    pool.extend(vw.get("dismisses_or_mocks", []))
    pool = [p for p in pool if p]
    return pool or ["everyday topics"]


class _ComboCycler:
    """Cycles through every (topic, message_type) combo in shuffled order
    before repeating any — the actual diversity mechanism, not left to the
    model's own judgment of what counts as 'different enough'."""

    def __init__(self, master, seed=None):
        topics = _topic_pool(master)
        self._all = [(t, m) for t in topics for m in MESSAGE_TYPES]
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


def _gen_batch(master, persona_text, combos, temperature, avoid_topics=None):
    avoid_block = ""
    if avoid_topics:
        avoid_block = ("\nINSTRUCTIONS ALREADY USED (do not near-duplicate these):\n" +
                       "\n".join(f"- {t}" for t in avoid_topics))
    assignments = "\n".join(
        f"{i+1}. topic: {t} | message type: {m}" for i, (t, m) in enumerate(combos))
    user = (
        f"PROFILE:\n{json.dumps(master, ensure_ascii=False)}\n\n"
        f"PERSONA SUMMARY:\n{persona_text}\n"
        f"{avoid_block}\n\n"
        f"ASSIGNMENTS ({len(combos)} examples, one per line):\n{assignments}\n\n"
        f"Generate exactly {len(combos)} examples, one per assignment above, "
        f"in order. Return ONLY the JSON array."
    )
    raw = _complete(
        [{"role": "system", "content": GEN_SYSTEM},
         {"role": "user", "content": user}],
        temperature=temperature, max_tokens=2200,
    )
    text = raw.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                text = part
                break
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        arr = json.loads(text[start:end + 1])
    except Exception:
        try:
            arr = json.loads(re.sub(r",\s*([}\]])", r"\1", text[start:end + 1]))
        except Exception:
            return []
    combo_topics = {t.lower().strip() for t, _ in combos}
    out = []
    for ex in arr:
        if not isinstance(ex, dict):
            continue
        inst = str(ex.get("instruction", "")).strip()
        outp = str(ex.get("output", "")).strip()
        if not inst or not outp:
            continue
        # Non-thinking dataset: reject anything that snuck in a think block.
        if "<think" in outp.lower():
            continue
        # Reject the model leaking the internal (topic, message-type)
        # scaffolding into the fake instruction instead of writing an actual
        # message — e.g. "game design and generative/procedural systems" or
        # "<topic> | message type: <type>" showing up as the "instruction".
        inst_lower = inst.lower().strip()
        if "message type" in inst_lower or " | " in inst:
            continue
        if inst_lower in combo_topics:
            continue
        # Reject the model echoing its own instruction back verbatim inside
        # the answer (a real failure mode seen in testing, not hypothetical).
        if len(inst) > 25 and inst_lower in outp.lower():
            continue
        out.append({"instruction": inst, "input": "", "output": outp})
    return out


def _is_near_duplicate(text, existing, threshold=0.62):
    """Cheap similarity check (difflib ratio) against everything accepted so
    far. Catches both exact repeats and 'same sentence, different wrapper'."""
    t_norm = text.lower().strip()
    for e in existing:
        if difflib.SequenceMatcher(None, t_norm, e.lower().strip()).ratio() >= threshold:
            return True
    return False


def generate_dataset(persona_name, total=300, batch_size=8, temperature=0.9, fresh=False):
    """total is the TARGET FINAL SIZE of the file (not "how many to add this
    run") — by default this resumes/accumulates across calls, seeding the
    dedup pool from whatever's already in the file, so re-running after a
    run that stalled out early keeps building the dataset up instead of
    re-hitting the same wall from zero. Pass fresh=True to discard and
    restart instead."""
    if not _engine_alive():
        raise SystemExit(f"Engine not reachable at {ENGINE_URL} — start it first.")

    persona_dir = os.path.join(INPUT_ROOT, persona_name)
    master_path = os.path.join(persona_dir, "master_persona.json")
    persona_md_path = os.path.join(persona_dir, "persona.md")
    if not os.path.exists(master_path):
        raise SystemExit(f"No master_persona.json for '{persona_name}' — run 'build' first.")

    master = json.load(open(master_path, encoding="utf-8"))
    persona_text = (open(persona_md_path, encoding="utf-8").read()
                    if os.path.exists(persona_md_path) else "")

    os.makedirs(DATASET_ROOT, exist_ok=True)
    out_path = os.path.join(DATASET_ROOT, f"{persona_name}.jsonl")

    # Seed the "don't repeat this" pool with the profile's own voice-reference
    # quotes (so the very first batch already knows not to just quote them
    # back) plus whatever's already in the dataset file from a prior run.
    seen_instructions = []
    seen_outputs = list(master.get("sample_voice_lines", []))
    added = 0
    if not fresh and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                seen_instructions.append(row.get("instruction", ""))
                seen_outputs.append(row.get("output", ""))
                added += 1
        if added:
            print(f"  ↺ resuming — {added} existing examples loaded from {out_path}")

    cycler = _ComboCycler(master)
    print(f"  ℹ {len(cycler._all)} distinct (topic × message-type) combinations available")

    batches, empty_streak = 0, 0
    n_batches = max(1, (total - added + batch_size - 1) // batch_size)
    max_batches = n_batches * 3  # headroom to retry past duplicate-heavy batches

    with open(out_path, "a" if (added and not fresh) else "w", encoding="utf-8") as f:
        while added < total and batches < max_batches:
            this_batch = min(batch_size, total - added)
            batches += 1
            avoid = seen_instructions[-25:]  # keep the prompt bounded
            combos = cycler.draw(this_batch)
            examples = _gen_batch(master, persona_text, combos, temperature, avoid_topics=avoid)

            kept = 0
            for ex in examples:
                # Instruction dedup stays strict (0.62) -- topic variety is what
                # the combo cycler is there to guarantee. Output dedup is looser
                # (0.85): a short, terse voice legitimately reuses phrasing
                # across unrelated topics ("yeah idk" answering two different
                # questions isn't a duplicate example), so only reject outputs
                # that are near-literal repeats, not just similarly short.
                if (_is_near_duplicate(ex["instruction"], seen_instructions) or
                        _is_near_duplicate(ex["output"], seen_outputs, threshold=0.85)):
                    continue
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                seen_instructions.append(ex["instruction"])
                seen_outputs.append(ex["output"])
                added += 1
                kept += 1

            dropped = len(examples) - kept
            print(f"  batch {batches} → +{kept} kept, {dropped} duplicate(s) dropped (total {added}/{total})")

            if kept == 0:
                empty_streak += 1
                if empty_streak >= 3:
                    print("  ⚠ 3 batches in a row produced nothing new — stopping early")
                    break
            else:
                empty_streak = 0

    print(f"✓ {added} examples written → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────
#  5. TRAIN — hand off to pipeline.py (separate, heavier step)
# ─────────────────────────────────────────────────────────────

def train_persona(persona_name, model_name="Qwen/Qwen3-8B", epochs=3, rank=16):
    import pipeline
    dataset_path = os.path.join(DATASET_ROOT, f"{persona_name}.jsonl")
    if not os.path.exists(dataset_path):
        raise SystemExit(f"No dataset at {dataset_path} — run 'gendata' first.")

    config = {
        "model_name": model_name,
        "lora_name": persona_name,
        "dataset_path": dataset_path,
        "max_seq_length": 1024,
        "lora_rank": rank, "lora_alpha": rank * 2, "lora_dropout": 0.05,
        "num_epochs": epochs, "batch_size": 2, "grad_accum": 4,
        "learning_rate": 1.5e-4, "logging_steps": 5, "save_steps": 200,
    }
    print(f"▶ Training LoRA '{persona_name}' on {model_name} ({dataset_path})")
    pipeline.reset_state(persona_name)
    stopped = pipeline.run_training(config)
    if stopped:
        raise SystemExit("Training did not complete.")

    print("▶ Exporting adapter to GGUF...")
    # _run_export_adapter has no return value — its result lives in
    # pipeline.prep_state (NOT pipeline.gguf_state, which belongs to the
    # separate run_gguf_conversion/merged-base-model path and is untouched
    # here — reading it was a real bug that silently registered characters
    # with an empty adapter_gguf even when export had just failed).
    pipeline._run_export_adapter(persona_name, persona_name, "")
    export = dict(pipeline.prep_state)
    adapter_path = export.get("output_path") or ""
    if export.get("status") != "done" or not adapter_path:
        print(f"✗ Adapter export failed: {export.get('error', 'unknown error')}")
        print(f"  LoRA weights are still safe in loras/{persona_name}/ — fix the export "
             f"issue, then re-run just the export (no need to retrain).")
    else:
        print(f"✓ Adapter GGUF ready: {adapter_path}")

    import store
    from voice import default_voice_for, has_cloned_voice
    persona_dir = os.path.join(INPUT_ROOT, persona_name)
    persona_md_path = os.path.join(persona_dir, "persona.md")
    persona_text = (open(persona_md_path, encoding="utf-8").read()
                    if os.path.exists(persona_md_path) else "")
    # Prefer a real cloned voice (voice_samples/<name>/*.wav) if one exists;
    # otherwise fall back to a deterministic Kokoro preset.
    voice = f"clone:{persona_name}" if has_cloned_voice(persona_name) else default_voice_for(persona_name)
    cfg = store.save_character({
        "name": persona_name,
        "persona": persona_text,
        "base_gguf": "gguf_output/Qwen3-8B-q8-base-Q8_0.gguf",
        "adapter_gguf": adapter_path,
        "voice": voice,
    })
    print(f"✓ Character registered: characters/{cfg['id']}/config.json"
         + ("" if adapter_path else "  ⚠ WITHOUT a working adapter — text generation "
                                     "will just be the base model until export is fixed and re-run."))
    return cfg


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="ingest docs → master_persona.json + persona.md")
    p_build.add_argument("persona_name")
    p_build.add_argument("--limit", type=int, default=None, help="only process first N chunks (testing)")
    p_build.add_argument("--sample", type=int, default=None,
                         help="randomly sample N chunks from across the whole corpus (for large sources)")

    p_gen = sub.add_parser("gendata", help="master_persona.json → datasets/<name>.jsonl")
    p_gen.add_argument("persona_name")
    p_gen.add_argument("-n", "--count", type=int, default=300, help="target final dataset size")
    p_gen.add_argument("--batch-size", type=int, default=8)
    p_gen.add_argument("--fresh", action="store_true", help="discard existing dataset instead of resuming/accumulating")

    p_train = sub.add_parser("train", help="run QLoRA training + GGUF export + register character")
    p_train.add_argument("persona_name")
    p_train.add_argument("--model", default="Qwen/Qwen3-8B")
    p_train.add_argument("--epochs", type=int, default=3)

    p_all = sub.add_parser("all", help="build + gendata (stops before train)")
    p_all.add_argument("persona_name")
    p_all.add_argument("-n", "--count", type=int, default=300)

    args = ap.parse_args()

    if args.cmd == "build":
        build_master_persona(args.persona_name, limit=args.limit, sample=args.sample)
    elif args.cmd == "gendata":
        generate_dataset(args.persona_name, total=args.count, batch_size=args.batch_size, fresh=args.fresh)
    elif args.cmd == "train":
        train_persona(args.persona_name, model_name=args.model, epochs=args.epochs)
    elif args.cmd == "all":
        build_master_persona(args.persona_name)
        generate_dataset(args.persona_name, total=args.count)
        print(f"\nNext: python persona_forge.py train {args.persona_name}")


if __name__ == "__main__":
    main()
