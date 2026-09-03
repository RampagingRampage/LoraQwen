"""
app.py — LoraQwen Forge. One backend for the whole thing.

Everything that used to be a separate CLI script or a Tkinter window is a route
here: persona distillation, dataset generation and cleanup, QLoRA training, GGUF
export, evaluation, DPO ranking, voice recording, chat, and the NEXUS feed.

    python src/app.py        → http://127.0.0.1:5000
"""

import os
import sys
import json
import time
import atexit
import threading
import subprocess

# core/ holds the application package; importing it puts core/ and the project
# root on sys.path so the modules inside can keep importing each other by name.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from flask import (Flask, request, jsonify, send_from_directory,
                   Response, stream_with_context, abort)

import config
import store
import dataset as ds
import generate as gen
import evaluate as ev
import voicelab
import feed as feedmod
import pipeline
import persona as pf
import voice as voicemod
from engine import MANAGER, build_system_prompt

config.ensure_dirs()

app = Flask(__name__, static_folder=config.STATIC_DIR, static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024   # voice clips and corpora


# ─────────────────────────────────────────────────────────────
#  UTIL
# ─────────────────────────────────────────────────────────────

def body():
    return request.get_json(silent=True) or {}


def fail(msg, code=400):
    return jsonify({"error": str(msg)}), code


def safe_name(value, what="name"):
    """Reject anything that isn't a plain single path segment. Every route that
    turns user input into a filesystem path goes through this."""
    v = os.path.basename(str(value or "").strip())
    if not v or v in (".", "..") or "/" in v or "\\" in v:
        raise ValueError(f"Invalid {what}: {value!r}")
    return v


def guard(fn):
    """Turn expected failures into clean JSON instead of a 500 + HTML page."""
    from functools import wraps

    @wraps(fn)
    def inner(*a, **kw):
        try:
            return fn(*a, **kw)
        except (ValueError, KeyError) as e:
            return fail(e, 400)
        except FileNotFoundError as e:
            return fail(e, 404)
        except RuntimeError as e:
            return fail(e, 409)
        except Exception as e:
            app.logger.exception("unhandled error in %s", fn.__name__)
            return fail(f"{type(e).__name__}: {e}", 500)
    return inner


# ─────────────────────────────────────────────────────────────
#  SHELL + BOOTSTRAP
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(config.STATIC_DIR, "index.html")


@app.route("/api/bootstrap")
@guard
def bootstrap():
    """Everything the front end needs on first paint, in one round trip."""
    return jsonify({
        "characters":   store.list_characters(),
        "base_ggufs":   _list_ggufs(base_only=True),
        "adapters":     _list_ggufs(base_only=False),
        "datasets":     ds.list_datasets(),
        "personas":     _list_personas(),
        "loras":        _list_loras(),
        "models":       _list_models(),
        "voices":       {"kokoro": voicemod.KOKORO_VOICES,
                         "clones": voicelab.list_voices()},
        "engine":       MANAGER.state(),
        "train_presets": config.TRAIN_PRESETS,
        "train_defaults": config.TRAIN_DEFAULTS,
        "defaults": {
            "chat": {"temperature": config.CHAT_TEMPERATURE,
                     "max_tokens": config.CHAT_MAX_TOKENS,
                     "top_p": config.CHAT_TOP_P},
            "generation": {"total": config.GEN_TOTAL, "batch_size": config.GEN_BATCH,
                           "temperature": config.GEN_TEMPERATURE,
                           "timeout_s": config.GEN_TIMEOUT_S,
                           "dedupe_threshold": config.GEN_DEDUPE,
                           "supervise_pct": config.GEN_SUPERVISE_PCT},
            "engine": {"ctx": config.DEFAULT_CTX, "ngl": config.DEFAULT_NGL,
                       "parallel": config.DEFAULT_PARALLEL,
                       "port": config.ENGINE_PORT},
            "lmstudio_url": config.LMSTUDIO_URL,
        },
        "feed_settings": feedmod.settings,
        "project_root": config.PROJECT_ROOT,
    })


def _list_ggufs(base_only=True):
    out = []
    if not os.path.isdir(config.GGUF_DIR):
        return out
    for f in sorted(os.listdir(config.GGUF_DIR)):
        if not f.lower().endswith(".gguf"):
            continue
        is_adapter = "-adapter-" in f.lower() or "adapter" in f.lower()
        if base_only and is_adapter:
            continue
        if not base_only and not is_adapter:
            continue
        p = os.path.join(config.GGUF_DIR, f)
        out.append({"name": f, "bytes": os.path.getsize(p),
                    "modified": os.path.getmtime(p)})
    return out


def _list_loras():
    out = []
    if not os.path.isdir(config.LORAS_DIR):
        return out
    for d in sorted(os.listdir(config.LORAS_DIR)):
        p = os.path.join(config.LORAS_DIR, d)
        if not os.path.isdir(p):
            continue
        cfg_path = os.path.join(p, "adapter_config.json")
        if not os.path.exists(cfg_path):
            continue
        meta = {}
        try:
            with open(cfg_path, encoding="utf-8") as f:
                raw = json.load(f)
            meta = {"rank": raw.get("r"), "alpha": raw.get("lora_alpha"),
                    "base": raw.get("base_model_name_or_path"),
                    "targets": raw.get("target_modules")}
        except (OSError, json.JSONDecodeError):
            pass
        ckpts = sorted(x for x in os.listdir(p) if x.startswith("checkpoint-"))
        out.append({"name": d, "modified": os.path.getmtime(p),
                    "checkpoints": ckpts, **meta})
    return out


def _list_models():
    out = []
    if not os.path.isdir(config.MODELS_DIR):
        return out
    for d in sorted(os.listdir(config.MODELS_DIR)):
        p = os.path.join(config.MODELS_DIR, d)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
            out.append({"name": d.replace("_", "/", 1), "dir": d})
    return out


def _list_personas():
    out = []
    if not os.path.isdir(config.TRAINING_INPUT):
        return out
    for d in sorted(os.listdir(config.TRAINING_INPUT)):
        p = os.path.join(config.TRAINING_INPUT, d)
        if not os.path.isdir(p) or d.startswith("_"):
            continue
        docs = [f for f in os.listdir(p)
                if f.lower().endswith((".txt", ".json", ".md", ".jsonl"))
                and not f.startswith("_")
                and f not in ("master_persona.json", "persona.md")]
        out.append({
            "name": d,
            "has_master": os.path.exists(os.path.join(p, "master_persona.json")),
            "has_md": os.path.exists(os.path.join(p, "persona.md")),
            "documents": docs,
        })
    return out


# ─────────────────────────────────────────────────────────────
#  ENGINE
# ─────────────────────────────────────────────────────────────

@app.route("/api/engine/status")
def engine_status():
    return jsonify(MANAGER.state())


@app.route("/api/engine/start", methods=["POST"])
@guard
def engine_start():
    d = body()
    # basename() the incoming filename: it used to be joined onto gguf_output/
    # unchecked, so an absolute path or ../ escaped the directory entirely.
    base = os.path.join(config.GGUF_DIR, safe_name(d.get("base_gguf"), "base GGUF"))
    if not os.path.exists(base):
        return fail(f"Base GGUF not found: {os.path.basename(base)}", 404)

    wanted = d.get("char_ids")
    adapters = []
    for c in store.list_characters():
        if wanted is not None and c["id"] not in wanted:
            continue
        if c.get("adapter_gguf"):
            adapters.append((c["id"], config.abspath(c["adapter_gguf"])))

    ok = MANAGER.start(
        base, adapters,
        llama_hint=d.get("llama_cpp_path", config.LLAMA_CPP_DIR),
        ctx=int(d.get("ctx", config.DEFAULT_CTX)),
        ngl=int(d.get("ngl", config.DEFAULT_NGL)),
        port=int(d.get("port", config.ENGINE_PORT)),
        parallel=int(d.get("parallel", config.DEFAULT_PARALLEL)))
    return jsonify({"ok": ok, "state": MANAGER.state()})


@app.route("/api/engine/stop", methods=["POST"])
@guard
def engine_stop():
    MANAGER.stop()
    return jsonify({"ok": True, "state": MANAGER.state()})


@app.route("/api/engine/kill_orphans", methods=["POST"])
@guard
def engine_kill_orphans():
    """llama-server.exe outlives its parent on Windows. This is the in-app
    equivalent of `taskkill /F /IM llama-server.exe /T`."""
    killed = _kill_llama_servers()
    MANAGER.stop()
    return jsonify({"ok": True, "killed": killed})


def _kill_llama_servers():
    if sys.platform != "win32":
        return subprocess.run(["pkill", "-f", "llama-server"],
                              capture_output=True).returncode == 0
    r = subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe", "/T"],
                       capture_output=True, text=True)
    return "SUCCESS" in (r.stdout or "").upper()


atexit.register(lambda: MANAGER.stop())


# ─────────────────────────────────────────────────────────────
#  CHARACTERS + MEMORIES
# ─────────────────────────────────────────────────────────────

@app.route("/api/characters", methods=["GET"])
def characters_list():
    return jsonify(store.list_characters())


@app.route("/api/characters", methods=["POST"])
@guard
def characters_create():
    d = body()
    if not (d.get("name") or "").strip():
        return fail("A name is required.")
    return jsonify(store.save_character(d))


@app.route("/api/characters/<cid>", methods=["GET"])
@guard
def character_get(cid):
    char = store.get_character(safe_name(cid, "character id"))
    if not char:
        return fail("No such character", 404)
    return jsonify(char)


@app.route("/api/characters/<cid>", methods=["PUT"])
@guard
def character_update(cid):
    cid = safe_name(cid, "character id")
    existing = store.get_character(cid)
    if not existing:
        return fail("No such character", 404)
    existing.update(body())
    existing["id"] = cid
    return jsonify(store.save_character(existing))


@app.route("/api/characters/<cid>", methods=["DELETE"])
@guard
def character_delete(cid):
    return jsonify({"ok": store.delete_character(safe_name(cid, "character id"))})


@app.route("/api/characters/<cid>/memories", methods=["GET"])
@guard
def memories_list(cid):
    return jsonify(store.get_memories(safe_name(cid, "character id")))


@app.route("/api/characters/<cid>/memories", methods=["POST"])
@guard
def memories_add(cid):
    d = body()
    text = (d.get("text") or "").strip()
    if not text:
        return fail("Memory text is required.")
    return jsonify(store.add_memory(safe_name(cid, "character id"), text,
                                    bool(d.get("enabled", True))))


@app.route("/api/characters/<cid>/memories/<mid>", methods=["PATCH"])
@guard
def memories_update(cid, mid):
    d = body()
    res = store.update_memory(safe_name(cid, "character id"), mid,
                              d.get("text"), d.get("enabled"))
    return jsonify(res) if res else fail("No such memory", 404)


@app.route("/api/characters/<cid>/memories/<mid>", methods=["DELETE"])
@guard
def memories_delete(cid, mid):
    return jsonify({"ok": store.delete_memory(safe_name(cid, "character id"), mid)})


@app.route("/api/characters/<cid>/memories/purge_auto", methods=["POST"])
@guard
def memories_purge_auto(cid):
    """Clear the 'I once said: ...' memories the feed's old always-on
    auto-memorize wrote back into RAG."""
    removed = feedmod.purge_auto_memories(safe_name(cid, "character id"))
    return jsonify({"ok": True, "removed": removed})


# ─────────────────────────────────────────────────────────────
#  CHAT  (the Inference tab)
# ─────────────────────────────────────────────────────────────

@app.route("/api/characters/<cid>/chats", methods=["GET"])
@guard
def chats_list(cid):
    chats = store.get_chats(safe_name(cid, "character id"))
    # The sidebar only needs titles and counts — sending every message of every
    # chat on each poll is a lot of payload for nothing.
    return jsonify([{
        "id": c["id"], "title": c.get("title", "New chat"),
        "created": c.get("created", 0), "updated": c.get("updated", c.get("created", 0)),
        "messages": len(c.get("messages", [])),
        "settings": c.get("settings", {}),
    } for c in sorted(chats, key=lambda c: -(c.get("updated") or c.get("created") or 0))])


@app.route("/api/characters/<cid>/chats/<chid>", methods=["GET"])
@guard
def chat_get(cid, chid):
    for c in store.get_chats(safe_name(cid, "character id")):
        if c["id"] == chid:
            return jsonify(c)
    return fail("No such chat", 404)


@app.route("/api/characters/<cid>/chats", methods=["POST"])
@guard
def chat_new(cid):
    d = body()
    return jsonify(store.new_chat(safe_name(cid, "character id"),
                                  (d.get("title") or "New chat").strip(),
                                  d.get("settings") or {}))


@app.route("/api/characters/<cid>/chats/<chid>", methods=["PUT"])
@guard
def chat_save(cid, chid):
    d = body()
    return jsonify(store.save_chat(safe_name(cid, "character id"), chid,
                                   d.get("messages", []), d.get("title"),
                                   d.get("settings")))


@app.route("/api/characters/<cid>/chats/<chid>", methods=["DELETE"])
@guard
def chat_delete(cid, chid):
    return jsonify({"ok": store.delete_chat(safe_name(cid, "character id"), chid)})


@app.route("/api/chat", methods=["POST"])
def chat_stream():
    """Streamed inference over SSE.

    Per-request overrides for system prompt, temperature, top_p, max tokens,
    tool use, reasoning and memory injection — so the Inference tab can change
    any of them per chat without touching the character's saved defaults.
    """
    d = body()
    char_id = (d.get("char_id") or "").strip() or None
    messages = d.get("messages") or []
    if not messages:
        return fail("No messages provided.")
    if not MANAGER.is_ready():
        return fail("Engine not running — start it from the Engine tab.", 409)

    char = store.get_character(char_id) if char_id else None
    tools_enabled = bool(d.get("tools_enabled", False))
    use_memories = bool(d.get("use_memories", True))
    enable_thinking = bool(d.get("enable_thinking", False))

    # An explicit system prompt from the UI wins; otherwise build one from the
    # character's persona plus (optionally) its enabled memories.
    override = (d.get("system_prompt") or "").strip()
    if override:
        system = override
        if tools_enabled:
            system = build_system_prompt(override, [], True)
    else:
        persona = (char or {}).get("persona", "")
        mems = store.enabled_memories(char_id)[:8] if (char_id and use_memories) else []
        system = build_system_prompt(persona, mems, tools_enabled)

    convo = [{"role": "system", "content": system}]
    convo += [m for m in messages if m.get("role") in ("user", "assistant")]

    temperature = float(d.get("temperature", config.CHAT_TEMPERATURE))
    max_tokens = int(d.get("max_tokens", config.CHAT_MAX_TOKENS))

    def events():
        try:
            for ev_obj in MANAGER.chat_stream(convo, char_id,
                                              temperature=temperature,
                                              max_tokens=max_tokens,
                                              tools_enabled=tools_enabled,
                                              enable_thinking=enable_thinking):
                yield f"data: {json.dumps(ev_obj, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return Response(stream_with_context(events()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/chat/title", methods=["POST"])
@guard
def chat_title():
    """Name a chat from its first exchange, so the sidebar isn't ten rows of
    'New chat'."""
    d = body()
    first = (d.get("text") or "").strip()
    if not first:
        return fail("Nothing to title.")
    if not MANAGER.is_ready():
        return jsonify({"title": first[:40]})
    try:
        title = MANAGER.complete(
            [{"role": "system", "content":
              "Reply with a 2-5 word title for this conversation. "
              "No quotes, no punctuation at the end, no preamble."},
             {"role": "user", "content": first[:600]}],
            char_id=None, temperature=0.3, max_tokens=20, enable_thinking=False)
        title = title.strip().strip('"').split("\n")[0][:60]
        return jsonify({"title": title or first[:40]})
    except Exception:
        return jsonify({"title": first[:40]})


# ─────────────────────────────────────────────────────────────
#  PERSONA
# ─────────────────────────────────────────────────────────────

persona_state = {"running": False, "status": "idle", "done": 0, "total": 0,
                 "name": None, "log": [], "error": None}


@app.route("/api/personas")
@guard
def personas_list():
    return jsonify(_list_personas())


@app.route("/api/persona/<name>", methods=["GET"])
@guard
def persona_get(name):
    name = safe_name(name, "persona")
    master, md = gen.load_persona(name)
    pdir = os.path.join(config.TRAINING_INPUT, name)
    docs = []
    if os.path.isdir(pdir):
        for f in sorted(os.listdir(pdir)):
            p = os.path.join(pdir, f)
            if os.path.isfile(p) and not f.startswith("_"):
                docs.append({"file": f, "bytes": os.path.getsize(p)})
    return jsonify({"name": name, "master": master, "persona_md": md,
                    "documents": docs, "topics": gen.topic_pool(name)})


@app.route("/api/persona/<name>", methods=["PUT"])
@guard
def persona_save(name):
    """Save hand-edits to the distilled profile or the short prompt version."""
    name = safe_name(name, "persona")
    d = body()
    pdir = os.path.join(config.TRAINING_INPUT, name)
    os.makedirs(pdir, exist_ok=True)
    if "master" in d:
        pf._atomic_write_json(os.path.join(pdir, "master_persona.json"), d["master"])
    if "persona_md" in d:
        with open(os.path.join(pdir, "persona.md"), "w", encoding="utf-8") as f:
            f.write(d["persona_md"])
    return jsonify({"ok": True})


@app.route("/api/persona/<name>/upload", methods=["POST"])
@guard
def persona_upload(name):
    name = safe_name(name, "persona")
    pdir = os.path.join(config.TRAINING_INPUT, name)
    os.makedirs(pdir, exist_ok=True)
    saved = []
    for f in request.files.getlist("files"):
        fn = safe_name(f.filename, "filename")
        if not fn.lower().endswith((".txt", ".json", ".jsonl", ".md", ".csv")):
            continue
        dest = os.path.join(pdir, fn)
        f.save(dest)
        saved.append({"file": fn, "bytes": os.path.getsize(dest)})
    if not saved:
        return fail("No usable files — accepted types are .txt .json .jsonl .md .csv")
    return jsonify({"ok": True, "saved": saved})


@app.route("/api/persona/<name>/books", methods=["GET"])
@guard
def books_list(name):
    import bookextract as be
    return jsonify(be.list_books(safe_name(name, "persona")))


@app.route("/api/persona/<name>/books", methods=["POST"])
@guard
def books_upload(name):
    """Upload the text of a book you already have the rights to — a file you
    export from an ebook you own, or paste yourself. This route never fetches
    anything on its own; it only stores what you send it."""
    import bookextract as be
    name = safe_name(name, "persona")
    f = request.files.get("file")
    text = request.form.get("text")
    filename = request.form.get("filename") or (f.filename if f else "book.txt")
    if f:
        text = f.read().decode("utf-8", errors="replace")
    if not text or not text.strip():
        return fail("No text provided.")
    saved = be.save_book(name, safe_name(filename, "filename"), text)
    return jsonify({"ok": True, "file": saved, "chars": len(text)})


@app.route("/api/persona/<name>/books/<path:filename>", methods=["DELETE"])
@guard
def books_delete(name, filename):
    import bookextract as be
    return jsonify({"ok": be.delete_book(safe_name(name, "persona"), filename)})


@app.route("/api/persona/<name>/books/<path:filename>/preview", methods=["POST"])
@guard
def books_preview(name, filename):
    import bookextract as be
    d = body()
    return jsonify(be.preview(
        safe_name(name, "persona"), filename,
        target_chars=int(d.get("chunk_chars", 3000)),
        max_chars=int(d.get("chunk_chars", 3000)) + 1500))


@app.route("/api/persona/<name>/filter", methods=["POST"])
@guard
def persona_filter(name):
    """Preview or apply the URL/phone/short-line filter on a raw export."""
    name = safe_name(name, "persona")
    d = body()
    src = os.path.join(config.TRAINING_INPUT, name, safe_name(d.get("file"), "file"))
    if not os.path.exists(src):
        return fail(f"No such file: {os.path.basename(src)}", 404)

    import textfilter as tf
    min_chars = int(d.get("min_chars", 12))
    kept, dropped, samples = [], 0, []
    with open(src, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if tf.URL_RE.search(s) or tf.PHONE_RE.search(s) or len(s) < min_chars:
                dropped += 1
                if len(samples) < 8:
                    samples.append(s[:160])
                continue
            kept.append(s)

    result = {"kept": len(kept), "dropped": dropped,
              "total": len(kept) + dropped, "dropped_samples": samples}
    if d.get("apply"):
        out_name = os.path.splitext(os.path.basename(src))[0] + "_filtered.txt"
        out = os.path.join(config.TRAINING_INPUT, name, out_name)
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(kept))
        result["written"] = out_name
    return jsonify(result)


@app.route("/api/persona/<name>/build", methods=["POST"])
@guard
def persona_build(name):
    """Distill raw documents into master_persona.json + persona.md."""
    name = safe_name(name, "persona")
    if persona_state["running"]:
        return fail("A persona build is already running.", 409)
    if not pf._engine_alive():
        return fail("Engine not running — the distiller needs it.", 409)

    d = body()
    limit = d.get("limit")
    sample = d.get("sample")

    def run():
        persona_state.update({"running": True, "status": "distilling", "name": name,
                              "done": 0, "total": 0, "log": [], "error": None})
        try:
            chunks = pf.gather_chunks(name)
            persona_state["total"] = len(chunks)
            master = pf.build_master_persona(
                name, limit=int(limit) if limit else None,
                sample=int(sample) if sample else None)
            pf.write_persona_string(master)
            persona_state["status"] = "done"
            persona_state["done"] = persona_state["total"]
        except Exception as e:
            persona_state["status"] = "error"
            persona_state["error"] = str(e)
        finally:
            persona_state["running"] = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/persona/build_status")
def persona_build_status():
    return jsonify(persona_state)


# ─────────────────────────────────────────────────────────────
#  DATASETS
# ─────────────────────────────────────────────────────────────

@app.route("/api/datasets")
@guard
def datasets_list():
    return jsonify(ds.list_datasets())


@app.route("/api/dataset/<name>")
@guard
def dataset_read(name):
    name = safe_name(name, "dataset")
    rows, bad = ds.read_rows(name)
    offset = request.args.get("offset", 0, type=int)
    limit = request.args.get("limit", 200, type=int)
    q = (request.args.get("q") or "").strip().lower()
    if q:
        rows = [r for r in rows
                if q in r["instruction"].lower() or q in r["output"].lower()]
    page = rows[offset:offset + limit]
    for r in page:
        r.pop("_raw", None)
    return jsonify({"name": name, "total": len(rows), "offset": offset,
                    "rows": page, "bad_lines": bad,
                    "queue": ds.queue_stats(name)})


@app.route("/api/dataset/<name>/row/<int:idx>", methods=["PUT"])
@guard
def dataset_row_update(name, idx):
    name = safe_name(name, "dataset")
    rows, _ = ds.read_rows(name)
    if not 0 <= idx < len(rows):
        return fail("Row out of range", 404)
    d = body()
    for key in ("instruction", "input", "output", "reasoning"):
        if key in d:
            rows[idx][key] = d[key]
    ds.write_rows(name, rows, tag="row-edit")
    return jsonify({"ok": True, "row": {k: v for k, v in rows[idx].items() if k != "_raw"}})


@app.route("/api/dataset/<name>/rows", methods=["DELETE"])
@guard
def dataset_rows_delete(name):
    """Bulk delete by index — the Data tab's multi-select."""
    name = safe_name(name, "dataset")
    idxs = set(int(i) for i in (body().get("indices") or []))
    if not idxs:
        return fail("No rows selected.")
    rows, _ = ds.read_rows(name)
    keep = [r for i, r in enumerate(rows) if i not in idxs]
    res = ds.write_rows(name, keep, tag="delete")
    return jsonify({"ok": True, "removed": len(rows) - len(keep), **res})


@app.route("/api/dataset/<name>/regenerate", methods=["POST"])
@guard
def dataset_regenerate(name):
    """Rewrite one row's answer, keeping its prompt."""
    name = safe_name(name, "dataset")
    d = body()
    idx = int(d.get("index", -1))
    rows, _ = ds.read_rows(name)
    if not 0 <= idx < len(rows):
        return fail("Row out of range", 404)
    if not MANAGER.is_ready() and d.get("source", "engine") == "engine":
        return fail("Engine not running.", 409)

    text = gen.regenerate_one(
        name, rows[idx]["instruction"],
        source=d.get("source", "engine"),
        lmstudio_url=d.get("lmstudio_url"), model=d.get("model"),
        temperature=float(d.get("temperature", 0.9)))
    if d.get("apply", False):
        rows[idx]["output"] = text
        ds.write_rows(name, rows, tag="regenerate")
    return jsonify({"ok": True, "output": text, "applied": bool(d.get("apply"))})


@app.route("/api/dataset/<name>/analyze", methods=["POST"])
@guard
def dataset_analyze(name):
    name = safe_name(name, "dataset")
    d = body()
    persona_text = ""
    if d.get("persona"):
        _, persona_text = gen.load_persona(safe_name(d["persona"], "persona"))
    return jsonify(ds.analyze(
        name,
        dupe_threshold=float(d.get("dupe_threshold", config.GEN_DEDUPE)),
        max_seq_length=int(d.get("max_seq_length", 1024)),
        persona_text=persona_text))


@app.route("/api/dataset/<name>/clean", methods=["POST"])
@guard
def dataset_clean(name):
    name = safe_name(name, "dataset")
    d = body()
    return jsonify(ds.apply_clean(
        name, d.get("ops") or [],
        dupe_threshold=float(d.get("dupe_threshold", config.GEN_DEDUPE)),
        max_seq_length=int(d.get("max_seq_length", 1024))))


@app.route("/api/dataset/<name>/snapshots")
@guard
def dataset_snapshots(name):
    return jsonify(ds.list_snapshots(safe_name(name, "dataset")))


@app.route("/api/dataset/<name>/restore", methods=["POST"])
@guard
def dataset_restore(name):
    name = safe_name(name, "dataset")
    ds.restore_snapshot(name, body().get("snapshot", ""))
    return jsonify({"ok": True})


@app.route("/api/dataset/<name>/export")
@guard
def dataset_export(name):
    name = safe_name(name, "dataset")
    return send_from_directory(config.DATASETS_DIR, f"{name}.jsonl", as_attachment=True)


@app.route("/api/dataset/create", methods=["POST"])
@guard
def dataset_create():
    name = safe_name(body().get("name"), "dataset")
    path = ds.dataset_path(name)
    if os.path.exists(path):
        return fail(f"'{name}' already exists.", 409)
    open(path, "w", encoding="utf-8").close()
    return jsonify({"ok": True, "name": name})


@app.route("/api/dataset/<name>", methods=["DELETE"])
@guard
def dataset_delete(name):
    name = safe_name(name, "dataset")
    path = ds.dataset_path(name)
    if not os.path.exists(path):
        return fail("No such dataset", 404)
    ds._snapshot(path, "pre-delete")   # deletable, but never unrecoverable
    os.remove(path)
    return jsonify({"ok": True})


@app.route("/api/dataset/import", methods=["POST"])
@guard
def dataset_import():
    """Bring in a .jsonl / .json / .csv, normalizing whatever shape it's in."""
    name = safe_name(request.form.get("name") or "imported", "dataset")
    f = request.files.get("file")
    if not f:
        return fail("No file uploaded.")
    raw = f.read().decode("utf-8", errors="replace")
    fname = (f.filename or "").lower()
    records = []

    if fname.endswith(".csv"):
        import csv, io
        for row in csv.DictReader(io.StringIO(raw)):
            rec = ds._normalize(dict(row), 0)
            if rec["instruction"] and rec["output"]:
                records.append(ds._to_record(rec))
    elif fname.endswith(".json"):
        data = json.loads(raw)
        for obj in (data if isinstance(data, list) else [data]):
            rec = ds._normalize(obj, 0)
            if rec["instruction"] and rec["output"]:
                records.append(ds._to_record(rec))
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = ds._normalize(json.loads(line), 0)
            except json.JSONDecodeError:
                continue
            if rec["instruction"] and rec["output"]:
                records.append(ds._to_record(rec))

    if not records:
        return fail("Nothing usable in that file — expected instruction/output pairs.")
    if request.form.get("replace") == "true":
        ds.write_rows(name, [ds._normalize(r, i) for i, r in enumerate(records)],
                      tag="import-replace")
    else:
        ds.append_rows(name, records)
    return jsonify({"ok": True, "name": name, "imported": len(records)})


# ─────────────────────────────────────────────────────────────
#  GENERATION + SUPERVISION
# ─────────────────────────────────────────────────────────────

@app.route("/api/generate/start", methods=["POST"])
@guard
def generate_start():
    d = body()
    src = d.get("source", "engine")
    book_src = d.get("book_source", "engine")
    if (src == "engine" or (src == "book" and book_src == "engine")) and not MANAGER.is_ready():
        return fail("Engine not running — start it, or switch the source to LM Studio.", 409)
    if src == "book":
        import bookextract as be
        if not d.get("book_file"):
            return fail("Pick a book to extract from.")
        if not (d.get("protagonist") or "").strip():
            return fail("Name the protagonist — the extractor needs to know whose lines to pull.")
        try:
            be.load_book(d.get("persona_name") or d.get("name"), d["book_file"])
        except FileNotFoundError:
            return fail(f"No such book: {d['book_file']}", 404)
    gen.start(d)
    return jsonify({"ok": True})


@app.route("/api/generate/status")
def generate_status():
    return jsonify(gen.gen_state)


@app.route("/api/generate/stop", methods=["POST"])
@guard
def generate_stop():
    return jsonify({"ok": gen.stop()})


@app.route("/api/generate/probe", methods=["POST"])
@guard
def generate_probe():
    """Is LM Studio (or any OpenAI-compatible endpoint) up, and what's loaded?"""
    return jsonify(gen.probe(body().get("url") or config.LMSTUDIO_URL))


@app.route("/api/generate/claude/<name>")
@guard
def generate_claude_spec(name):
    name = safe_name(name, "dataset")
    p = gen.claude_request_path(name)
    if not os.path.exists(p):
        return fail("No request spec written yet — start a run with source 'claude'.", 404)
    return jsonify({
        "request_path": config.rel(p),
        "response_path": config.rel(os.path.join(config.GEN_RESPONSES, f"{name}.jsonl")),
        "response_ready": os.path.exists(os.path.join(config.GEN_RESPONSES, f"{name}.jsonl")),
    })


@app.route("/api/generate/claude/<name>/collect", methods=["POST"])
@guard
def generate_claude_collect(name):
    d = body()
    return jsonify(gen.collect_claude(safe_name(name, "dataset"),
                                      supervise_pct=int(d.get("supervise_pct", 0))))


@app.route("/api/review/<name>")
@guard
def review_queue(name):
    name = safe_name(name, "dataset")
    items = [i for i in ds.load_queue(name) if i["status"] == "pending"]
    return jsonify({"items": items, "stats": ds.queue_stats(name)})


@app.route("/api/review/<name>/decide", methods=["POST"])
@guard
def review_decide(name):
    name = safe_name(name, "dataset")
    d = body()
    action = d.get("action")

    # "Regenerate" doesn't decide anything — it swaps the candidate in place
    # and leaves the item pending for another look.
    if action == "regenerate":
        for it in ds.load_queue(name):
            if it["id"] == d.get("id"):
                rec = it["record"]
                instr = rec.get("instruction") or (
                    rec.get("messages", [{}])[0].get("content", ""))
                text = gen.regenerate_one(name, instr,
                                          source=d.get("source", "engine"),
                                          temperature=float(d.get("temperature", 0.9)))
                return jsonify({"ok": True, "output": text})
        return fail("No such review item", 404)

    item = ds.queue_decide(name, d.get("id"), action, d.get("record"))
    return jsonify({"ok": True, "item": item, "stats": ds.queue_stats(name)})


@app.route("/api/review/<name>/clear", methods=["POST"])
@guard
def review_clear(name):
    return jsonify({"pending": ds.queue_clear_decided(safe_name(name, "dataset"))})


# ─────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────

@app.route("/api/train/start", methods=["POST"])
@guard
def train_start():
    d = body()
    if pipeline.training_state.get("running"):
        return fail("Training is already running.", 409)

    lora_name = safe_name(d.get("lora_name"), "adapter name")
    dataset_name = safe_name(d.get("dataset") or lora_name, "dataset")
    dataset_path = ds.dataset_path(dataset_name)
    if not os.path.exists(dataset_path):
        return fail(f"Dataset not found: {dataset_name}.jsonl", 404)

    cfg = dict(config.TRAIN_DEFAULTS)
    cfg.update({k: v for k, v in d.items() if k in cfg or k in
                ("resume_from_checkpoint", "llama_cpp_path")})
    cfg["lora_name"] = lora_name
    cfg["dataset_path"] = dataset_path
    for k in ("max_seq_length", "lora_rank", "lora_alpha", "num_epochs",
              "batch_size", "grad_accum", "logging_steps", "save_steps",
              "eval_split_pct", "seed"):
        cfg[k] = int(cfg[k])
    for k in ("lora_dropout", "learning_rate", "warmup_ratio"):
        cfg[k] = float(cfg[k])

    # Training runs in a child process — that is the only way to reliably get
    # every byte of VRAM back on Windows afterwards.
    threading.Thread(target=_train_and_archive, args=(cfg,), daemon=True).start()
    return jsonify({"ok": True, "config": cfg})


def _train_and_archive(cfg):
    try:
        pipeline.run_training(cfg)
    finally:
        try:
            snap = pipeline._read_status_file() or dict(pipeline.training_state)
            if snap.get("status") == "done":
                ev.save_training_run(cfg["lora_name"], snap)
        except Exception as e:
            print(f"[train] could not archive run: {e}")


@app.route("/api/train/status")
def train_status():
    ipc = pipeline._read_status_file() or {}
    live = dict(pipeline.training_state)
    # The IPC file is authoritative while a child process owns the run; the
    # in-memory dict covers the in-process fallback path.
    merged = {**live, **{k: v for k, v in ipc.items() if v is not None}}
    merged["gguf_export"] = dict(pipeline.prep_state)
    merged["generation"] = {"running": gen.is_running(), "status": gen.gen_state["status"]}
    return jsonify(merged)


@app.route("/api/train/stop", methods=["POST"])
@guard
def train_stop():
    pipeline.stop_event.set()
    try:
        open(pipeline.IPC_STOP, "w").close()
    except OSError:
        pass
    return jsonify({"ok": True})


@app.route("/api/train/runs/<name>")
@guard
def train_runs(name):
    return jsonify(ev.training_runs(safe_name(name, "adapter name")))


@app.route("/api/train/dpo", methods=["POST"])
@guard
def train_dpo():
    """DPO refinement on top of an existing SFT adapter."""
    d = body()
    name = safe_name(d.get("name"), "adapter name")
    pairs = os.path.join(config.DPO_DIR, f"{name}_dpo_pairs.jsonl")
    if not os.path.exists(pairs):
        return fail(f"No preference pairs yet — rank some responses first "
                    f"(expected {config.rel(pairs)}).", 404)
    if not os.path.isdir(os.path.join(config.LORAS_DIR, name)):
        return fail(f"No SFT adapter at loras/{name}/ to refine. Train one first.", 404)
    if pipeline.training_state.get("running"):
        return fail("Training is already running.", 409)

    import dpo as dpomod

    def run():
        try:
            dpomod.main_programmatic(name, d) if hasattr(dpomod, "main_programmatic") \
                else _run_dpo_subprocess(name)
        except Exception as e:
            print(f"[dpo] failed: {e}")

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True})


def _run_dpo_subprocess(name):
    """dpo.py was written as a CLI. Rather than restructure a working training
    script, run it the way it expects — in its own process, which also gets us
    the same VRAM reclamation the SFT path relies on."""
    script = os.path.join(config.PROJECT_ROOT, "core", "dpo.py")
    subprocess.run([sys.executable, script, name], cwd=config.PROJECT_ROOT)


@app.route("/api/loras")
@guard
def loras_list():
    return jsonify(_list_loras())


@app.route("/api/loras/<name>", methods=["DELETE"])
@guard
def lora_delete(name):
    import shutil
    p = os.path.join(config.LORAS_DIR, safe_name(name, "adapter name"))
    if not os.path.isdir(p):
        return fail("No such adapter", 404)
    shutil.rmtree(p, ignore_errors=True)
    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────
#  GGUF + MODEL DOWNLOAD
# ─────────────────────────────────────────────────────────────

@app.route("/api/gguf/export", methods=["POST"])
@guard
def gguf_export():
    d = body()
    name = safe_name(d.get("lora_name"), "adapter name")
    if not os.path.isdir(os.path.join(config.LORAS_DIR, name)):
        return fail(f"No adapter at loras/{name}/", 404)
    # _run_export_adapter appends "-adapter-{outtype}.gguf" itself -- passing
    # an out_name that already ends in "-adapter" (the old default here) doubled
    # it up into "<name>-adapter-adapter-f16.gguf". Bare name is what it wants.
    threading.Thread(
        target=pipeline._run_export_adapter,
        args=(name, d.get("out_name") or name,
              d.get("llama_cpp_path", config.LLAMA_CPP_DIR),
              d.get("outtype", "f16")),
        daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/gguf/prepare_base", methods=["POST"])
@guard
def gguf_prepare_base():
    d = body()
    model = d.get("model_name") or config.DEFAULT_MODEL
    threading.Thread(
        target=pipeline._run_prepare_base,
        args=(model, d.get("out_name") or model.split("/")[-1],
              d.get("llama_cpp_path", config.LLAMA_CPP_DIR),
              d.get("quantize", "f16")),
        daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/gguf/status")
def gguf_status():
    return jsonify(dict(pipeline.prep_state))


@app.route("/api/model/download", methods=["POST"])
@guard
def model_download():
    name = (body().get("model_name") or config.DEFAULT_MODEL).strip()
    threading.Thread(target=pipeline.run_model_download, args=(name,), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/model/status")
def model_status():
    return jsonify(dict(pipeline.model_dl_state))


@app.route("/api/register", methods=["POST"])
@guard
def register_character():
    """Turn a trained adapter into a usable character in one step."""
    d = body()
    name = safe_name(d.get("name"), "name")
    adapter = os.path.join(config.GGUF_DIR, safe_name(
        d.get("adapter_gguf") or f"{name}-adapter-f16.gguf", "adapter"))
    if not os.path.exists(adapter):
        return fail(f"No adapter GGUF at {config.rel(adapter)} — export it first.", 404)

    _, persona_md = gen.load_persona(name)
    existing = next((c for c in store.list_characters()
                     if c.get("name", "").lower() == name.lower()), None)
    char = {
        "id": (existing or {}).get("id"),
        "name": name,
        "persona": d.get("persona") or (existing or {}).get("persona") or persona_md,
        "base_gguf": d.get("base_gguf") or (existing or {}).get("base_gguf", ""),
        "adapter_gguf": adapter,
        "voice": d.get("voice") or (existing or {}).get("voice")
                 or (f"clone:{name}" if voicemod.has_cloned_voice(name) else ""),
        "created": (existing or {}).get("created"),
    }
    return jsonify(store.save_character(char))


# ─────────────────────────────────────────────────────────────
#  EVALUATE
# ─────────────────────────────────────────────────────────────

@app.route("/api/eval/compare", methods=["POST"])
@guard
def eval_compare():
    d = body()
    if not (d.get("prompt") or "").strip():
        return fail("A prompt is required.")
    return jsonify(ev.compare(
        d["prompt"], d.get("char_ids") or [],
        temperature=float(d.get("temperature", 0.8)),
        max_tokens=int(d.get("max_tokens", 200)),
        include_base=bool(d.get("include_base", True)),
        use_persona=bool(d.get("use_persona", True))))


@app.route("/api/eval/style", methods=["POST"])
@guard
def eval_style():
    d = body()
    return jsonify(ev.style_report(
        safe_name(d.get("char_id"), "character id"),
        safe_name(d.get("dataset"), "dataset"),
        prompts=d.get("prompts"),
        temperature=float(d.get("temperature", 0.8)),
        n=d.get("n")))


@app.route("/api/eval/memorization", methods=["POST"])
@guard
def eval_memorization():
    d = body()
    return jsonify(ev.memorization_check(
        safe_name(d.get("char_id"), "character id"),
        safe_name(d.get("dataset"), "dataset"),
        threshold=float(d.get("threshold", 0.82)),
        n=int(d.get("n", 12))))


@app.route("/api/eval/regression/prompts", methods=["GET", "POST"])
@guard
def eval_regression_prompts():
    if request.method == "POST":
        return jsonify({"prompts": ev.save_regression_prompts(body().get("prompts") or [])})
    return jsonify({"prompts": ev.load_regression_prompts()})


@app.route("/api/eval/regression/run", methods=["POST"])
@guard
def eval_regression_run():
    d = body()
    return jsonify(ev.run_regression(
        safe_name(d.get("char_id"), "character id"),
        temperature=float(d.get("temperature", 0.8)),
        label=d.get("label")))


@app.route("/api/eval/regression/<cid>")
@guard
def eval_regression_history(cid):
    return jsonify(ev.list_runs(safe_name(cid, "character id")))


# ─────────────────────────────────────────────────────────────
#  DPO RANKING
# ─────────────────────────────────────────────────────────────

@app.route("/api/dpo/<name>/stats")
@guard
def dpo_stats(name):
    name = safe_name(name, "name")
    prefs = os.path.join(config.DPO_DIR, f"{name}_preferences.jsonl")
    pairs = os.path.join(config.DPO_DIR, f"{name}_dpo_pairs.jsonl")

    def count(p):
        if not os.path.exists(p):
            return 0
        with open(p, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    return jsonify({"rounds": count(prefs), "pairs": count(pairs),
                    "ready": count(pairs) >= 20})


@app.route("/api/dpo/<name>/candidates", methods=["POST"])
@guard
def dpo_candidates(name):
    """Four answers to one training prompt at spread temperatures — real
    diversity to choose between, not four samples at the same setting."""
    name = safe_name(name, "name")
    if not MANAGER.is_ready():
        return fail("Engine not running — load this character's adapter first.", 409)

    d = body()
    char_id = d.get("char_id")
    char = store.get_character(char_id) if char_id else None
    if not char:
        return fail("Pick the character whose adapter is loaded.", 400)

    rows, _ = ds.read_rows(name)
    prompts = [r["instruction"] for r in rows if r["instruction"].strip()]
    if not prompts:
        return fail(f"Dataset '{name}' has no prompts to rank.", 404)

    import random
    prompt = d.get("prompt") or random.choice(prompts)
    temps = d.get("temperatures") or [0.6, 0.8, 1.0, 1.15]
    candidates = []
    for t in temps:
        try:
            candidates.append({
                "temperature": t,
                "text": ev._gen(prompt, char_id, char.get("persona", ""),
                                temperature=float(t),
                                max_tokens=int(d.get("max_tokens", 200))).strip(),
            })
        except Exception as e:
            candidates.append({"temperature": t, "text": "", "error": str(e)})
    return jsonify({"prompt": prompt, "candidates": candidates})


@app.route("/api/dpo/<name>/pick", methods=["POST"])
@guard
def dpo_pick(name):
    """Record one preference. Writes both the raw pick and the derived
    chosen/rejected pairs TRL's DPOTrainer consumes."""
    name = safe_name(name, "name")
    d = body()
    prompt = (d.get("prompt") or "").strip()
    candidates = d.get("candidates") or []
    chosen = (d.get("chosen") or "").strip()
    if not prompt or not chosen:
        return fail("A prompt and a chosen response are required.")

    os.makedirs(config.DPO_DIR, exist_ok=True)
    with open(os.path.join(config.DPO_DIR, f"{name}_preferences.jsonl"),
              "a", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": prompt, "candidates": candidates,
                            "chosen": chosen, "custom": bool(d.get("custom")),
                            "ts": time.time()}, ensure_ascii=False) + "\n")

    written = 0
    with open(os.path.join(config.DPO_DIR, f"{name}_dpo_pairs.jsonl"),
              "a", encoding="utf-8") as f:
        for c in candidates:
            text = (c.get("text") or "").strip()
            if not text or text == chosen:
                continue
            f.write(json.dumps({"prompt": prompt, "chosen": chosen,
                                "rejected": text}, ensure_ascii=False) + "\n")
            written += 1
    return jsonify({"ok": True, "pairs_written": written})


# ─────────────────────────────────────────────────────────────
#  VOICE
# ─────────────────────────────────────────────────────────────

@app.route("/api/voice/prompts")
def voice_prompts():
    return jsonify(voicelab.prompts())


@app.route("/api/voice/voices")
@guard
def voice_voices():
    return jsonify({"kokoro": voicemod.KOKORO_VOICES,
                    "clones": voicelab.list_voices()})


@app.route("/api/voice/samples/<name>")
@guard
def voice_samples(name):
    name = safe_name(name, "voice")
    return jsonify({"name": name, "clips": voicelab.list_clips(name)})


@app.route("/api/voice/clip/<name>/<path:filename>")
@guard
def voice_clip(name, filename):
    d = os.path.join(config.VOICE_SAMPLES, safe_name(name, "voice"))
    return send_from_directory(d, safe_name(filename, "filename"))


@app.route("/api/voice/samples/<name>", methods=["POST"])
@guard
def voice_upload(name):
    name = safe_name(name, "voice")
    f = request.files.get("file")
    if not f:
        return fail("No audio uploaded.")
    label = safe_name(request.form.get("label") or "clip", "label")
    ext = os.path.splitext(f.filename or "")[1] or ".webm"
    return jsonify(voicelab.save_upload(name, label, f.read(), ext))


@app.route("/api/voice/samples/<name>/<path:filename>", methods=["DELETE"])
@guard
def voice_delete(name, filename):
    return jsonify({"ok": voicelab.delete_clip(safe_name(name, "voice"),
                                               safe_name(filename, "filename"))})


@app.route("/api/voice/samples/<name>/refs", methods=["PUT"])
@guard
def voice_refs(name):
    return jsonify({"clips": voicelab.set_refs(safe_name(name, "voice"),
                                               body().get("clips") or [])})


@app.route("/api/chat/speak", methods=["POST"])
@guard
def chat_speak():
    """Synthesize one chat reply in the character's own voice.

    Used by the Inference tab -- separate from /api/voice/preview because it
    resolves the speaker from the character (its assigned clone/preset,
    falling back to the same deterministic Kokoro assignment the feed uses)
    rather than taking an explicit speaker name.
    """
    d = body()
    text = (d.get("text") or "").strip()
    if not text:
        return fail("Nothing to speak.")
    char_id = d.get("char_id")
    char = store.get_character(char_id) if char_id else None
    speaker = (char or {}).get("voice") or voicemod.default_voice_for(char_id or "")
    os.makedirs(config.AUDIO_DIR, exist_ok=True)
    base = os.path.join(config.AUDIO_DIR, f"chat_{int(time.time()*1000)}")
    path = voicemod.synthesize_to_file_auto(text, speaker, base)
    if not path:
        return fail("Synthesis produced no audio.", 500)
    return jsonify({"url": "/static/audio/" + os.path.basename(path), "speaker": speaker})


@app.route("/api/voice/preview", methods=["POST"])
@guard
def voice_preview():
    """Synthesize a line so you can hear a preset or a clone before committing
    a character to it."""
    d = body()
    text = (d.get("text") or "alright, let's see if this actually sounds right.").strip()
    speaker = (d.get("speaker") or "af_sarah").strip()
    os.makedirs(config.AUDIO_DIR, exist_ok=True)
    base = os.path.join(config.AUDIO_DIR, f"preview_{int(time.time()*1000)}")
    path = voicemod.synthesize_to_file_auto(text, speaker, base)
    if not path:
        return fail("Synthesis produced no audio.", 500)
    return jsonify({"url": "/static/audio/" + os.path.basename(path)})


# ─────────────────────────────────────────────────────────────
#  FEED
# ─────────────────────────────────────────────────────────────

@app.route("/api/feed/state")
def feed_state_route():
    since = request.args.get("since", 0, type=float)
    with feedmod.feed_lock:
        return jsonify({
            "posts": [p for p in feedmod.feed_state["posts"] if p["ts"] > since],
            "total_posts": len(feedmod.feed_state["posts"]),
            "active_bots": feedmod.feed_state["active_bots"],
            "is_thinking": feedmod.feed_state["is_thinking"],
            "is_posting": feedmod.feed_state["is_posting"],
            "current_thinker": feedmod.feed_state["current_thinker"],
            "thoughts": dict(feedmod.feed_state["thoughts"]),
        })


@app.route("/api/feed/bots", methods=["POST"])
@guard
def feed_bots():
    with feedmod.feed_lock:
        feedmod.feed_state["active_bots"] = body().get("bot_ids", [])
        feedmod.feed_state["turn_index"] = 0
        feedmod.feed_state["priority_queue"] = []
    feedmod.save()
    return jsonify({"ok": True})


@app.route("/api/feed/step", methods=["POST"])
@guard
def feed_step():
    if not MANAGER.is_ready():
        return fail("Engine not running.", 409)
    with feedmod.feed_lock:
        if not feedmod.feed_state["active_bots"]:
            return fail("No residents selected.")
        if feedmod.feed_state["is_thinking"] or feedmod.feed_state["is_posting"]:
            return fail("A bot is already mid-turn.", 409)
    threading.Thread(target=feedmod.generate_bot_post, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/feed/human", methods=["POST"])
@guard
def feed_human():
    d = body()
    text = (d.get("text") or "").strip()
    if not text:
        return fail("Empty post.")
    return jsonify({"ok": True,
                    "post": feedmod.add_human_post(text, d.get("username") or "you")})


@app.route("/api/feed/inject", methods=["POST"])
@guard
def feed_inject():
    topic = (body().get("topic") or "").strip()
    if not topic:
        return fail("No topic provided.")
    with feedmod.feed_lock:
        feedmod.feed_state["topic_injection"] = topic
    return jsonify({"ok": True})


@app.route("/api/feed/react", methods=["POST"])
@guard
def feed_react():
    d = body()
    if d.get("reaction") not in ("fire", "think", "disagree", "eye"):
        return fail("Invalid reaction.")
    res = feedmod.react(d.get("post_id"), d["reaction"])
    return jsonify({"ok": True, "reactions": res}) if res else fail("Post not found", 404)


@app.route("/api/feed/thoughts")
def feed_thoughts():
    with feedmod.feed_lock:
        return jsonify(dict(feedmod.feed_state["thoughts"]))


@app.route("/api/feed/settings", methods=["GET", "POST"])
@guard
def feed_settings():
    if request.method == "POST":
        feedmod.settings.update({k: v for k, v in body().items()
                                 if k in feedmod.settings})
        feedmod.save()
    return jsonify(feedmod.settings)


@app.route("/api/feed/clear", methods=["POST"])
@guard
def feed_clear():
    return jsonify({"ok": feedmod.clear()})


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    feedmod.load()
    feedmod.gc_audio()
    print("\n" + "=" * 52)
    print("  LoraQwen Forge")
    print(f"  http://{config.WEB_HOST}:{config.WEB_PORT}")
    print("=" * 52 + "\n")
    # threaded=True is load-bearing, not a nicety: a single post generation is
    # several seconds of LLM + TTS, and without it that blocks every other
    # request including the status polls the UI depends on.
    app.run(host=config.WEB_HOST, port=config.WEB_PORT,
            debug=False, threaded=True, use_reloader=False)
