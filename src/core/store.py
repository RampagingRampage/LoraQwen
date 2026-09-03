"""
store.py — unified persistence layer for LoRA Forge (merges former rag.py).

Merges store.py (character/chat/legacy-memory CRUD) and rag.py (semantic
vector store) into a single coherent module. The split was a historical
accident: store.py was written first with simple list-based memories, then
rag.py was bolted on later to power cognition.py — and the two stores were
never reconciled. This module:

  • Keeps the character and chat APIs identical (store.py drop-in)
  • Keeps the Rag class API identical (rag.py drop-in)
  • Removes the legacy memories.json sidecar (store.get_memories /
    store.add_memory etc.) — the RAG store is the one source of truth for
    all memory content, including the old character-level memories
  • Adds a thin CharacterMemory helper that lets callers interact with a
    per-character slice of the RAG store using the old memories.json API
    shape, so cognition.py's _should_rest() check still works without
    modification

Migration: on first load, if a character directory still has memories.json,
its contents are automatically ingested into the RAG store (with
meta.char_id set) and the file is renamed memories.json.migrated so the
import runs exactly once.

Usage (drop-in for both old modules):

    import memory as store          # full store.py API
    import memory as rag_mod        # Rag class still here
    from memory import RAG          # shared singleton (path configurable)

    # Character memories via RAG:
    mems = store.get_memories(char_id)          # -> [{id,text,enabled,created}]
    store.add_memory(char_id, "text")
    store.enabled_memories(char_id)             # -> [str, ...]

    # Raw RAG:
    RAG.add("some fact", {"type":"memory","category":"research"})
    RAG.search("related query", k=5)
"""

from __future__ import annotations

import os, json, math, re, hashlib, threading, time, uuid, shutil
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────

import config as _cfg

# Absolute, from core/config.py (.env), so every entry point agrees on where
# characters live regardless of the working directory it was launched from.
ROOT     = _cfg.CHARACTERS_DIR                   # character directory root
RAG_PATH = os.path.join(ROOT, "_rag_store.json") # single shared vector store

_DIM = 384   # MiniLM embedding dimension; hash-fallback uses same width


# ─────────────────────────────────────────────────────────────
#  EMBEDDING BACKEND
# ─────────────────────────────────────────────────────────────

class _Embedder:
    """
    Tries sentence-transformers (all-MiniLM-L6-v2) for real embeddings;
    falls back to a hashed bag-of-words vector so everything works out of
    the box without GPU or extra installs.
    """

    def __init__(self):
        self.model = None
        self.kind  = "hash"
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.kind  = "sbert"
            print("✓ memory: sentence-transformers (all-MiniLM-L6-v2) loaded")
        except Exception as e:
            print(
                f"ℹ memory: sentence-transformers not available ({e.__class__.__name__}); "
                f"using hashed bag-of-words fallback. "
                f"`pip install sentence-transformers` for better recall."
            )

    def embed(self, text: str) -> list[float]:
        text = (text or "").strip()
        if self.kind == "sbert":
            v = self.model.encode([text], normalize_embeddings=True)[0]
            return [float(x) for x in v]
        # Hashed bag-of-words fallback
        vec = [0.0] * _DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % _DIM]           += 1.0
            vec[(h // _DIM) % _DIM] += 0.5
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / n for x in vec]


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ─────────────────────────────────────────────────────────────
#  RAG STORE  (rag.py drop-in)
# ─────────────────────────────────────────────────────────────

class Rag:
    """
    Semantic vector store with full CRUD, categories, and embedding.

    All items are stored as:
        {id, text, meta, vec, ts}

    meta is an arbitrary dict; the system uses:
        type     — "memory" | "injected_memory" | "nudge" | "web"
        category — free-form string for grouping
        char_id  — (new) optional character scope; enables per-character views
        enabled  — (new) boolean, mirrors old memories.json enabled flag
        source   — "ego" | "archivist" | "user" | "migration" etc.
    """

    def __init__(self, path: str = RAG_PATH):
        self.path  = path
        self._lock = threading.Lock()
        self._emb  = _Embedder()
        self.items = self._load()
        self._migrate_legacy_memories()

    # ── Persistence ─────────────────────────────────────────────────────────

    def _load(self) -> list:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    # ── Legacy migration ─────────────────────────────────────────────────────
    # Runs once at startup. Finds any memories.json files under ROOT, imports
    # them as RAG items with char_id set, then renames the file .migrated so
    # this path never runs again.

    def _migrate_legacy_memories(self):
        if not os.path.isdir(ROOT):
            return
        for subdir in Path(ROOT).iterdir():
            if not subdir.is_dir():
                continue
            mpath = subdir / "memories.json"
            if not mpath.exists():
                continue
            try:
                mems = json.loads(mpath.read_text(encoding="utf-8"))
                char_id = subdir.name
                imported = 0
                for m in mems:
                    text = (m.get("text") or "").strip()
                    if not text:
                        continue
                    # Skip if already imported (dedup by text content)
                    hits = self.search(text, k=1)
                    if hits and hits[0]["score"] > 0.97:
                        continue
                    self.add(text, {
                        "type":     "memory",
                        "category": "character",
                        "char_id":  char_id,
                        "enabled":  bool(m.get("enabled", True)),
                        "source":   "migration",
                        "orig_id":  m.get("id", ""),
                    })
                    imported += 1
                mpath.rename(mpath.with_suffix(".json.migrated"))
                if imported:
                    print(f"✓ memory: migrated {imported} legacy memories from {char_id}")
            except Exception as e:
                print(f"⚠ memory: migration failed for {subdir.name}: {e}")

    # ── Basic CRUD ───────────────────────────────────────────────────────────

    def count(self) -> int:
        return len(self.items)

    def add(self, text: str, meta: Optional[dict] = None) -> Optional[str]:
        text = (text or "").strip()
        if not text:
            return None
        with self._lock:
            rid = hashlib.md5((text + str(time.time())).encode()).hexdigest()[:12]
            self.items.append({
                "id":   rid,
                "text": text,
                "meta": meta or {},
                "vec":  self._emb.embed(text),
                "ts":   time.time(),
            })
            self._save()
            return rid

    def update(self, item_id: str, text: Optional[str] = None,
               meta: Optional[dict] = None) -> Optional[dict]:
        """Edit an existing item's text and/or meta. Re-embeds if text changes."""
        with self._lock:
            for it in self.items:
                if it.get("id") == item_id:
                    if text is not None:
                        text = text.strip()
                        if text:
                            it["text"] = text
                            it["vec"]  = self._emb.embed(text)
                    if meta is not None:
                        if isinstance(meta, dict):
                            it["meta"].update(meta)
                        else:
                            it["meta"] = meta
                    self._save()
                    return it
            return None

    def delete(self, item_id: str) -> bool:
        with self._lock:
            before = len(self.items)
            self.items = [it for it in self.items if it.get("id") != item_id]
            if len(self.items) < before:
                self._save()
                return True
            return False

    def get(self, item_id: str) -> Optional[dict]:
        """Return a single item by id (without vec), or None."""
        for it in self.items:
            if it.get("id") == item_id:
                return {k: v for k, v in it.items() if k != "vec"}
        return None

    def clear(self):
        with self._lock:
            self.items = []
            self._save()

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5,
               char_id: Optional[str] = None) -> list[dict]:
        """
        Semantic search. If char_id is given, restricts to items whose
        meta.char_id matches (enables per-character memory lanes).
        Returns [{id, text, meta, score, ts}], best first.
        """
        q = self._emb.embed(query)
        pool = self.items
        if char_id is not None:
            pool = [it for it in pool
                    if (it.get("meta") or {}).get("char_id") == char_id]
        scored = [
            {
                "id":    it["id"],
                "text":  it["text"],
                "meta":  it.get("meta", {}),
                "score": round(_cos(q, it["vec"]), 4),
                "ts":    it.get("ts", 0),
            }
            for it in pool
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:max(1, int(k))]

    # ── Category helpers ─────────────────────────────────────────────────────

    def get_categories(self) -> list[str]:
        """Return a sorted list of unique category strings across all items."""
        cats = set()
        for it in self.items:
            cat = (it.get("meta") or {}).get("category")
            if cat:
                cats.add(str(cat))
        return sorted(cats)

    def get_by_category(self, category: str,
                         char_id: Optional[str] = None) -> list[dict]:
        """
        Return all items (without vec) whose meta.category matches
        (case-insensitive). Optionally filter to a specific char_id.
        """
        cat_lower = (category or "").strip().lower()
        out = []
        for it in self.items:
            m = it.get("meta") or {}
            if m.get("category", "").strip().lower() != cat_lower:
                continue
            if char_id is not None and m.get("char_id") != char_id:
                continue
            out.append({k: v for k, v in it.items() if k != "vec"})
        out.sort(key=lambda x: x.get("ts", 0), reverse=True)
        return out

    def all_items(self, char_id: Optional[str] = None) -> list[dict]:
        """Return all items without vec, newest first. Optionally scoped to char_id."""
        with self._lock:
            pool = self.items
            if char_id is not None:
                pool = [it for it in pool
                        if (it.get("meta") or {}).get("char_id") == char_id]
            items = sorted(pool, key=lambda x: x.get("ts", 0), reverse=True)
            return [{k: v for k, v in it.items() if k != "vec"} for it in items]

    # ── Per-character views (new, used by CharacterMemory below) ────────────

    def get_char_memories(self, char_id: str) -> list[dict]:
        """Return items whose meta.char_id == char_id, shaped like old memories.json."""
        raw = self.all_items(char_id=char_id)
        return [
            {
                "id":      it["id"],
                "text":    it["text"],
                "enabled": bool((it.get("meta") or {}).get("enabled", True)),
                "created": it.get("ts", 0),
            }
            for it in raw
        ]

    def count_char_memories(self, char_id: str) -> int:
        return sum(
            1 for it in self.items
            if (it.get("meta") or {}).get("char_id") == char_id
        )


# ── Module-level singleton ───────────────────────────────────────────────────
# Lazily initialised so importing memory.py at module level doesn't force
# disk IO before the characters/ directory has been created.

_rag_instance: Optional[Rag] = None
_rag_lock = threading.Lock()


def _get_rag() -> Rag:
    global _rag_instance
    if _rag_instance is None:
        with _rag_lock:
            if _rag_instance is None:
                _rag_instance = Rag(RAG_PATH)
    return _rag_instance


# Public alias — cognition.py does `from memory import RAG` or `memory.RAG`.
# We use a property-like accessor at module level via __getattr__ (Python 3.7+).
def __getattr__(name):
    if name == "RAG":
        return _get_rag()
    raise AttributeError(name)


# ─────────────────────────────────────────────────────────────
#  SHARED PERSISTENCE HELPERS  (store.py internals)
# ─────────────────────────────────────────────────────────────

def _cdir(char_id: str) -> str:
    return os.path.join(ROOT, char_id)


def _read(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ─────────────────────────────────────────────────────────────
#  CHARACTERS  (store.py drop-in)
# ─────────────────────────────────────────────────────────────

def list_characters() -> list[dict]:
    Path(ROOT).mkdir(exist_ok=True)
    out = []
    for sub in sorted(Path(ROOT).iterdir(), key=lambda x: x.name):
        if sub.is_dir():
            # Go through get_character so the legacy absolute-path migration
            # runs here too, not only on individual reads.
            cfg = get_character(sub.name)
            if cfg:
                out.append(cfg)
    return out


def get_character(char_id: str) -> Optional[dict]:
    cfg = _read(os.path.join(_cdir(char_id), "config.json"), None)
    if cfg:
        # Migrate legacy absolute paths in place, once, on first read.
        changed = False
        for key in ("base_gguf", "adapter_gguf"):
            val = cfg.get(key) or ""
            if val and os.path.isabs(val):
                cfg[key] = _cfg.rel(val)
                changed = True
        if changed:
            _write(os.path.join(_cdir(char_id), "config.json"), cfg)
    return cfg


def save_character(data: dict) -> dict:
    cid = data.get("id") or _new_id()
    cfg = {
        "id":           cid,
        "name":         (data.get("name") or "Unnamed").strip(),
        "persona":      data.get("persona", "") or "",
        # Paths are stored RELATIVE to the project root. An absolute Windows
        # path baked in here breaks the character the moment the folder moves,
        # which is exactly what happened to the first trained adapter.
        "base_gguf":    _cfg.rel(data.get("base_gguf", "") or ""),
        "adapter_gguf": _cfg.rel(data.get("adapter_gguf", "") or ""),
        "voice":        data.get("voice", "") or "",  # Kokoro speaker, or clone:<name>
        "created":      data.get("created") or time.time(),
        "settings":     data.get("settings") or {},   # per-character chat defaults
    }
    _write(os.path.join(_cdir(cid), "config.json"), cfg)
    # ensure the chats sidecar exists (memories.json is gone; RAG is used instead)
    cpath = os.path.join(_cdir(cid), "chats.json")
    if not os.path.exists(cpath):
        _write(cpath, [])
    return cfg


def delete_character(char_id: str) -> bool:
    """Delete a character and purge its RAG memories."""
    d = _cdir(char_id)
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
        # purge from RAG store
        rag = _get_rag()
        to_delete = [
            it["id"] for it in rag.items
            if (it.get("meta") or {}).get("char_id") == char_id
        ]
        for rid in to_delete:
            rag.delete(rid)
        return True
    return False


# ─────────────────────────────────────────────────────────────
#  MEMORIES (backed by RAG — store.py drop-in API)
# ─────────────────────────────────────────────────────────────
# All functions below replace the old memories.json sidecar. They proxy
# through the RAG store, scoped to char_id via meta.char_id.

def get_memories(char_id: str) -> list[dict]:
    """Return [{id, text, enabled, created}] for a character, newest first."""
    return _get_rag().get_char_memories(char_id)


def add_memory(char_id: str, text: str, enabled: bool = True) -> dict:
    rag = _get_rag()
    rid = rag.add(text.strip(), {
        "type":     "memory",
        "category": "character",
        "char_id":  char_id,
        "enabled":  bool(enabled),
        "source":   "user",
    })
    return {
        "id":      rid,
        "text":    text.strip(),
        "enabled": bool(enabled),
        "created": time.time(),
    }


def update_memory(char_id: str, mem_id: str,
                  text: Optional[str] = None,
                  enabled: Optional[bool] = None) -> Optional[dict]:
    rag = _get_rag()
    # Validate ownership before mutating
    item = rag.get(mem_id)
    if not item:
        return None
    if (item.get("meta") or {}).get("char_id") != char_id:
        return None  # wrong owner — refuse silently
    meta_update = {}
    if enabled is not None:
        meta_update["enabled"] = bool(enabled)
    result = rag.update(mem_id, text=text, meta=meta_update or None)
    if result is None:
        return None
    return {
        "id":      result["id"],
        "text":    result["text"],
        "enabled": bool((result.get("meta") or {}).get("enabled", True)),
        "created": result.get("ts", 0),
    }


def delete_memory(char_id: str, mem_id: str) -> bool:
    rag = _get_rag()
    item = rag.get(mem_id)
    if not item:
        return False
    if (item.get("meta") or {}).get("char_id") != char_id:
        return False
    return rag.delete(mem_id)


def enabled_memories(char_id: str) -> list[str]:
    """Return text strings for all enabled memories of a character."""
    return [
        m["text"] for m in get_memories(char_id)
        if m.get("enabled", True)
    ]


# ─────────────────────────────────────────────────────────────
#  CHATS  (store.py drop-in — unchanged, still in JSON sidecars)
# ─────────────────────────────────────────────────────────────

def get_chats(char_id: str) -> list[dict]:
    return _read(os.path.join(_cdir(char_id), "chats.json"), [])


def _save_chats(char_id: str, chats: list):
    _write(os.path.join(_cdir(char_id), "chats.json"), chats)


def new_chat(char_id: str, title: str = "New chat", settings: Optional[dict] = None) -> dict:
    chats = get_chats(char_id)
    chat = {"id": _new_id(), "title": title, "created": time.time(),
            "updated": time.time(), "messages": [], "settings": settings or {}}
    chats.append(chat)
    _save_chats(char_id, chats)
    return chat


def save_chat(char_id: str, chat_id: str,
              messages: list, title: Optional[str] = None,
              settings: Optional[dict] = None) -> dict:
    chats = get_chats(char_id)
    for c in chats:
        if c["id"] == chat_id:
            c["messages"] = messages
            c["updated"] = time.time()
            if title is not None:
                c["title"] = title
            if settings is not None:
                c["settings"] = settings
            _save_chats(char_id, chats)
            return c
    chat = {
        "id": chat_id, "title": title or "New chat",
        "created": time.time(), "updated": time.time(),
        "messages": messages, "settings": settings or {},
    }
    chats.append(chat)
    _save_chats(char_id, chats)
    return chat


def delete_chat(char_id: str, chat_id: str) -> bool:
    chats = get_chats(char_id)
    new = [c for c in chats if c["id"] != chat_id]
    _save_chats(char_id, new)
    return len(new) != len(chats)


# ─────────────────────────────────────────────────────────────
#  COMPATIBILITY SHIM — legacy `from rag import Rag` imports
# ─────────────────────────────────────────────────────────────
# Any module that does `from rag import Rag` or `import rag; rag.Rag(...)`
# can be updated to `from memory import Rag` or `import memory as rag`.
# The class is exported at the top level; nothing else is required.