"""
core/bookextract.py — literal dialogue extraction from a book.

Different technique from generate.py's normal path. generate.py distills a
voice PROFILE from source text, then has the model INVENT new scenarios in
that voice — everything in the resulting dataset is synthetic. This module
instead walks real chapters and pulls real triplets straight out of the
prose: {instruction: what another character said or the situation, reasoning:
the protagonist's own narrated inner thought if the passage shows one,
output: the protagonist's actual line or action}.

Real prose as ground truth is higher-fidelity than synthetic generation for
capturing a subtle voice, but it comes with real costs this module is built
around:

  - only works cleanly on a FIRST-PERSON (or very close-third) narrator whose
    own thoughts are actually on the page — a third-person narrator has no
    native "inner thought in the protagonist's own words" to extract, and
    asking the model to invent one defeats the point of using real text at all
  - extracted triplets are real excerpts of the source, not paraphrases —
    this is built for PERSONAL, LOCAL training data from text you already
    have the right to use (a book you own), never for the tool to go fetch
    copyrighted text on its own, and the resulting dataset should not be
    redistributed
  - dialogue density varies wildly by book; expect tens to a few hundred
    clean triplets per novel after ambiguous attributions are filtered out,
    not thousands
  - real-book training data is exactly the case where the Evaluate tab's
    memorization check matters — run it after training

Chunking is chapter- and paragraph-aware (never mid-sentence), unlike
persona.py's fixed-character chunker, because a scene needs to stay whole for
an exchange to extract cleanly.
"""

import os
import re
import json
import time
import random
import threading

import config
import dataset as ds
import persona as pf
from generate import _salvage_json_array, _endpoint_for

BOOKS_SUBDIR = "_books"

CHAPTER_RE = re.compile(
    r"^\s*(chapter|part|book)\s+([ivxlcdm]+|\d+|one|two|three|four|five|six|seven|"
    r"eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)

EXTRACT_SYSTEM = (
    "You extract training examples from a novel excerpt for a dialogue dataset. "
    "The protagonist is {protagonist}. Find every distinct moment in this excerpt "
    "where {protagonist} speaks or clearly acts in response to something — another "
    "character's line, a situation, an event.\n\n"
    "For each one, produce an object with:\n"
    '  "instruction" — what {protagonist} is responding to, in 1-3 sentences: '
    "another character's dialogue (quote it if short, paraphrase if long) or the "
    "situation they're reacting to. Written as if someone else is describing the "
    "moment to {protagonist}, not as narration.\n"
    '  "reasoning" — {protagonist}\'s own inner thought or reaction AS NARRATED IN '
    "THE TEXT, if this excerpt actually shows one. Use the real wording where "
    "possible. Leave this an empty string if the passage doesn't show their "
    "interiority for this moment — do not invent one.\n"
    '  "output" — {protagonist}\'s actual line of dialogue or described action, '
    "taken directly from the text, as close to verbatim as the excerpt allows.\n\n"
    "Skip anything where the speaker is ambiguous, where {protagonist} doesn't "
    "appear, or where you'd have to guess rather than read. Fewer correct examples "
    "beats more uncertain ones. If nothing in this excerpt qualifies, return an "
    "empty array.\n\n"
    "Output ONLY a JSON array: "
    '[{{"instruction":"...","reasoning":"","output":"..."}}]. No prose, no fences.'
)


# ─────────────────────────────────────────────────────────────
#  STORAGE
# ─────────────────────────────────────────────────────────────

def _books_dir(persona_name):
    d = os.path.join(config.TRAINING_INPUT, os.path.basename(str(persona_name)), BOOKS_SUBDIR)
    os.makedirs(d, exist_ok=True)
    return d


def list_books(persona_name):
    d = _books_dir(persona_name)
    out = []
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith((".txt", ".md")):
            continue
        p = os.path.join(d, f)
        with open(p, encoding="utf-8", errors="replace") as fh:
            chars = len(fh.read())
        out.append({"file": f, "bytes": os.path.getsize(p), "chars": chars,
                    "chapters": len(CHAPTER_RE.findall(open(p, encoding="utf-8", errors="replace").read()))})
    return out


def save_book(persona_name, filename, text):
    fname = os.path.basename(str(filename))
    if not fname.lower().endswith((".txt", ".md")):
        fname += ".txt"
    path = os.path.join(_books_dir(persona_name), fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return fname


def delete_book(persona_name, filename):
    path = os.path.join(_books_dir(persona_name), os.path.basename(str(filename)))
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def load_book(persona_name, filename):
    path = os.path.join(_books_dir(persona_name), os.path.basename(str(filename)))
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such book: {filename}")
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────
#  CHUNKING — chapter-aware, paragraph-safe
# ─────────────────────────────────────────────────────────────

def split_chapters(text):
    """Split on chapter/part headers if present; otherwise the whole text is
    one chapter. Front matter before the first header (title page, table of
    contents) is dropped — it has no protagonist dialogue in it anyway."""
    marks = list(CHAPTER_RE.finditer(text))
    if not marks:
        return [text]
    chapters = []
    for i, m in enumerate(marks):
        start = m.start()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        chapters.append(text[start:end])
    return chapters


def chunk_chapter(chapter_text, target_chars=3000, max_chars=4500):
    """Paragraph-boundary chunks sized to comfortably hold a few full
    exchanges — big enough for context, small enough that the model isn't
    asked to track a whole chapter's cast at once."""
    paras = [p for p in re.split(r"\n\s*\n", chapter_text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) > max_chars:
            chunks.append(buf.strip())
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
        if len(buf) >= target_chars:
            chunks.append(buf.strip())
            buf = ""
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def preview(persona_name, filename, target_chars=3000, max_chars=4500):
    """Chapter/chunk counts without calling the model — lets the UI show what
    a run would actually touch before committing to it."""
    text = load_book(persona_name, filename)
    chapters = split_chapters(text)
    chunk_counts = [len(chunk_chapter(c, target_chars, max_chars)) for c in chapters]
    return {
        "chapters": len(chapters),
        "chunks": sum(chunk_counts),
        "chars": len(text),
        "chunks_per_chapter": chunk_counts[:40],
    }


# ─────────────────────────────────────────────────────────────
#  EXTRACTION
# ─────────────────────────────────────────────────────────────

def _extract_chunk(chunk_text, protagonist, url, model, temperature, timeout_s, max_tokens):
    sys_msg = EXTRACT_SYSTEM.format(protagonist=protagonist)
    raw = pf._complete(
        [{"role": "system", "content": sys_msg},
         {"role": "user", "content": f"EXCERPT:\n{chunk_text}"}],
        temperature=temperature, max_tokens=max_tokens, timeout=timeout_s,
        url=url, model=model)
    arr = _salvage_json_array(raw)
    out = []
    for ex in arr:
        if not isinstance(ex, dict):
            continue
        inst = str(ex.get("instruction", "")).strip()
        outp = str(ex.get("output", "")).strip()
        if not inst or not outp:
            continue
        out.append({"instruction": inst, "output": outp,
                    "reasoning": str(ex.get("reasoning", "")).strip()})
    return out


def run(dataset_name, persona_name, filename, protagonist, log, stop_event,
       state, source="engine", lmstudio_url=None, model=None,
       temperature=0.3, timeout_s=90, max_tokens=1800,
       target_chars=3000, max_chars=4500, dedupe_threshold=0.7,
       supervise_pct=0, max_chunks=None):
    """Runs synchronously in the caller's thread — generate.py drives this
    from its own background thread and mirrors progress into gen_state, the
    same as every other source, so the Data tab needs no separate UI for it.

    dedupe_threshold defaults higher than synthetic generation's 0.62: real
    prose naturally repeats a character's verbal tics, and that repetition is
    signal, not noise, here — only reject near-exact re-extraction of the
    same passage.
    """
    text = load_book(persona_name, filename)
    chapters = split_chapters(text)
    all_chunks = []
    for ch in chapters:
        all_chunks.extend(chunk_chapter(ch, target_chars, max_chars))
    if max_chunks:
        all_chunks = all_chunks[:int(max_chunks)]

    url = _endpoint_for(source, lmstudio_url)
    rng = random.Random()

    existing, _ = ds.read_rows(dataset_name)
    seen_outputs = [r["output"] for r in existing if r["output"]]

    state["batches"] = len(all_chunks)
    log(f"extracting {protagonist!r} from {filename} — {len(chapters)} chapters, "
        f"{len(all_chunks)} chunks · supervise {supervise_pct}%")

    for i, chunk in enumerate(all_chunks):
        if stop_event.is_set():
            log("stopped by user")
            state["status"] = "stopped"
            return
        state["batch"] = i + 1
        try:
            found = _extract_chunk(chunk, protagonist, url, model, temperature,
                                   timeout_s, max_tokens)
        except Exception as e:
            log(f"chunk {i+1}/{len(all_chunks)} failed: {e}")
            continue

        kept = 0
        for ex in found:
            out = ex["output"]
            if pf._is_near_duplicate(out, seen_outputs, dedupe_threshold):
                state["rejected"] += 1
                continue
            rec = {"instruction": ex["instruction"], "input": "", "output": out}
            if ex["reasoning"]:
                rec["reasoning"] = ex["reasoning"]
            seen_outputs.append(out)
            kept += 1

            if supervise_pct and rng.randint(1, 100) <= supervise_pct:
                ds.queue_add(dataset_name, [rec], meta={
                    "source": "book", "book": filename, "protagonist": protagonist,
                    "chunk": i + 1})
                state["held"] += 1
            else:
                ds.append_rows(dataset_name, [rec])
                state["added"] += 1

        log(f"chunk {i+1}/{len(all_chunks)} — extracted {kept}/{len(found)} "
            f"(dataset {state['added']}, held {state['held']})")

    if state["status"] != "stopped":
        state["status"] = "done"
        log(f"finished — {state['added']} written, {state['held']} awaiting review "
            f"from {len(all_chunks)} chunks of {filename}")
