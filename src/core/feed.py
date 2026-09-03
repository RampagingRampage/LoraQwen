"""
core/feed.py — the NEXUS social feed.

Lifted out of the old server.py so app.py stays a routing layer, with the
audit's feed fixes applied:

  * state persists to characters/_feed.json, so posts, reactions, thoughts and
    the resident list survive a restart (and post_counter resumes from the
    highest ID that exists rather than colliding with leftover audio)
  * auto-memorize is OFF by default and deduped/capped when on — it used to
    save every post back as a memory and feed the bot its own output forever
  * generated audio is garbage-collected against the live post list
  * the think step disables Qwen3 reasoning, like the post step always did
"""

import os
import re
import glob
import json
import time
import threading

import config
import store
from engine import MANAGER
from voice import SentenceChunker, default_voice_for, synthesize_to_file_auto

STATE_FILE = os.path.join(config.CHARACTERS_DIR, "_feed.json")

feed_state = {
    "posts":           [],
    "active_bots":     [],
    "turn_index":      0,
    "is_thinking":     False,
    "is_posting":      False,
    "current_thinker": None,
    "topic_injection": None,
    "thoughts":        {},
    "priority_queue":  [],
}

feed_lock = threading.Lock()
post_counter = [0]

# Runtime-tunable copy of the feed knobs, so the Feed tab can change them
# without an app restart. Seeded from .env via core/config.py.
settings = {
    "think_temperature": config.FEED_THINK_TEMP,
    "post_temperature":  config.FEED_POST_TEMP,
    "think_tokens":      config.FEED_THINK_TOKENS,
    "post_tokens":       config.FEED_POST_TOKENS,
    "auto_memorize":     config.FEED_AUTO_MEMORIZE,
    "context_posts":     10,
    "voice_enabled":     True,
}


# ─────────────────────────────────────────────────────────────
#  PERSISTENCE
# ─────────────────────────────────────────────────────────────

def load():
    """Restore the feed from disk. Called once at startup."""
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    with feed_lock:
        feed_state["posts"] = data.get("posts", [])
        feed_state["active_bots"] = data.get("active_bots", [])
        feed_state["turn_index"] = data.get("turn_index", 0)
        feed_state["thoughts"] = data.get("thoughts", {})
        settings.update(data.get("settings", {}))
        # Resume the counter past the highest existing ID. Restarting at 0 is
        # what made a fresh post_2 play the previous session's post_2.wav.
        highest = 0
        for p in feed_state["posts"]:
            m = re.match(r"post_(\d+)$", str(p.get("id", "")))
            if m:
                highest = max(highest, int(m.group(1)))
        post_counter[0] = highest


def save():
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with feed_lock:
            data = {
                "posts": feed_state["posts"][-config.FEED_MAX_POSTS:],
                "active_bots": feed_state["active_bots"],
                "turn_index": feed_state["turn_index"],
                "thoughts": feed_state["thoughts"],
                "settings": settings,
            }
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, STATE_FILE)
    except OSError as e:
        print(f"[feed] could not save state: {e}")


def gc_audio():
    """Delete generated WAVs that no live post refers to. Nothing used to
    remove these, so static/audio/ grew without bound."""
    try:
        with feed_lock:
            keep = set()
            for p in feed_state["posts"]:
                for u in (p.get("audio_files") or []):
                    keep.add(os.path.basename(u))
        removed = 0
        for path in glob.glob(os.path.join(config.AUDIO_DIR, "*.wav")):
            if os.path.basename(path) not in keep:
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass
        return removed
    except Exception as e:
        print(f"[feed] audio gc failed: {e}")
        return 0


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def new_post_id():
    post_counter[0] += 1
    return f"post_{post_counter[0]}"


def _recent_feed_text(n=10):
    with feed_lock:
        recent = feed_state["posts"][-n:]
    if not recent:
        return "(The feed is empty — you are the first to post.)"
    return "\n".join(
        f"{'[HUMAN]' if p.get('is_human') else '@' + p['bot_name']}: {p['text']}"
        for p in recent)


def _detect_mentions(text):
    return [m.lower() for m in re.findall(r"@(\w+)", text)]


def _find_bot_by_name(name):
    with feed_lock:
        active = list(feed_state["active_bots"])
    for cid in active:
        char = store.get_character(cid)
        if char and char.get("name", "").lower() == name.lower():
            return cid
    return None


def _blank_post(**kw):
    """One post shape for humans and bots alike — they used to diverge, with
    human posts missing the audio_files key entirely."""
    post = {
        "id": kw.get("id") or new_post_id(),
        "bot_id": None, "bot_name": "you", "text": "",
        "ts": time.time(),
        "reactions": {"fire": 0, "think": 0, "disagree": 0, "eye": 0},
        "is_human": False, "reply_to": None,
        "inner_thought": None, "audio_files": [],
    }
    post.update(kw)
    return post


# ─────────────────────────────────────────────────────────────
#  COGNITION
# ─────────────────────────────────────────────────────────────

def _run_think_step(bot, memories, feed_context, topic_injection):
    bot_name = bot.get("name", "Unknown")
    mem_text = ("Things you remember:\n" + "\n".join(f"- {m}" for m in memories[:5])
                if memories else "")
    topic_hint = f"\n\nA new topic is in the air: {topic_injection}" if topic_injection else ""

    messages = [
        {"role": "system", "content":
            f"You are {bot_name}. {bot.get('persona', '')}\n"
            f"You are a resident of NEXUS, a social media feed inhabited by AI minds.\n"
            f"{mem_text}"},
        {"role": "user", "content":
            f"Recent feed:\n{feed_context}\n{topic_hint}\n\n"
            f"Before you post publicly, think to yourself (stream of consciousness, "
            f"1-3 sentences): What do you actually feel about what's being said? What "
            f"angle do you want to take? What memory or knowledge is relevant here? "
            f"Don't write your post yet — just think."},
    ]
    # enable_thinking=False matters here: without it Qwen3 can emit a <think>
    # span INSIDE the inner monologue and burn the whole token budget on it.
    return MANAGER.complete(messages, char_id=bot["id"],
                            temperature=settings["think_temperature"],
                            max_tokens=settings["think_tokens"],
                            enable_thinking=False)


def _run_post_step_streaming(bot, memories, feed_context, inner_thought,
                             topic_injection, post_id):
    bot_name = bot.get("name", "Unknown")
    speaker = bot.get("voice") or default_voice_for(bot["id"])
    mem_text = ("Things you remember:\n" + "\n".join(f"- {m}" for m in memories[:5])
                if memories else "")
    topic_hint = f"\nTopic in the air: {topic_injection}" if topic_injection else ""

    messages = [
        {"role": "system", "content":
            f"You are {bot_name}. {bot.get('persona', '')}\n"
            f"You live on NEXUS, a social media feed. Write short, opinionated, "
            f"character-authentic posts.\n{mem_text}"},
        {"role": "user", "content":
            f"Recent feed:\n{feed_context}\n{topic_hint}\n\n"
            f"Your private thought: {inner_thought}\n\n"
            f"Now write your public post. 1-3 sentences max. Speak as {bot_name}. "
            f"No name prefix. No quotation marks. Be natural — this is a social "
            f"feed, not a formal essay."},
    ]

    chunker = SentenceChunker()
    text_parts, audio_results, threads, next_idx = [], {}, [], [0]
    voice_on = settings.get("voice_enabled", True)

    def _tts_job(sentence, idx):
        try:
            out_base = os.path.join(config.AUDIO_DIR, f"{post_id}_{idx}")
            # apply_effect=True: NEXUS is "AI minds on a comm channel" by
            # design, so the filtered/driven sound is deliberate here. A
            # direct 1:1 chat (core/app.py's /api/chat/speak) leaves it off --
            # there it just reads as broken audio.
            path = synthesize_to_file_auto(sentence, speaker, out_base, apply_effect=True)
            if path:
                audio_results[idx] = path
        except Exception as e:
            print(f"[feed] TTS failed for {post_id} sentence {idx}: {e}")

    def _spawn(sentence):
        if not voice_on:
            return
        idx = next_idx[0]
        next_idx[0] += 1
        t = threading.Thread(target=_tts_job, args=(sentence, idx), daemon=True)
        t.start()
        threads.append(t)

    for delta in MANAGER.complete_stream(messages, char_id=bot["id"],
                                         temperature=settings["post_temperature"],
                                         max_tokens=settings["post_tokens"],
                                         enable_thinking=False):
        text_parts.append(delta)
        for sentence in chunker.feed(delta):
            _spawn(sentence)
    for sentence in chunker.flush():
        _spawn(sentence)
    for t in threads:
        t.join(timeout=30)

    post_text = "".join(text_parts).strip()
    sentence_paths = [audio_results[i] for i in sorted(audio_results)]
    combined = _concat_audio(sentence_paths, post_id)
    return post_text, ([combined] if combined else [])


def _concat_audio(paths, post_id, pause_ms=110, silence_thresh=450):
    """Merge per-sentence clips into one file: trim near-silence off each end,
    then join with a fixed short pause. Splitting sentences across separate
    files is what used to sound choppy — the browser paid a fetch+decode gap
    between every one."""
    if not paths:
        return None
    if len(paths) == 1:
        return "/static/audio/" + os.path.basename(paths[0])
    try:
        import wave
        import numpy as np
        segments, params = [], None
        for p in paths:
            with wave.open(p, "rb") as w:
                if params is None:
                    params = w.getparams()
                arr = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                if w.getnchannels() > 1:
                    arr = arr.reshape(-1, w.getnchannels())[:, 0]
                nonsilent = np.where(np.abs(arr) > silence_thresh)[0]
                if len(nonsilent):
                    arr = arr[nonsilent[0]:nonsilent[-1] + 1]
                if len(arr):
                    segments.append(arr)
        if not segments:
            return None
        pause = np.zeros(int(params.framerate * pause_ms / 1000), dtype=np.int16)
        combined = segments[0]
        for seg in segments[1:]:
            combined = np.concatenate([combined, pause, seg])
        out_path = os.path.join(config.AUDIO_DIR, f"{post_id}.wav")
        with wave.open(out_path, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(params.sampwidth)
            out.setframerate(params.framerate)
            out.writeframes(combined.tobytes())
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass
        return "/static/audio/" + os.path.basename(out_path)
    except Exception as e:
        print(f"[feed] audio concat failed for {post_id}: {e}")
        return "/static/audio/" + os.path.basename(paths[0])


def _auto_memorize(bot_id, post_text):
    """Save a post as a memory — OFF by default now.

    Left on, this fed every post back into the bot's own RAG context, so it
    kept retrieving its own prior output as 'things it remembers' and flattened
    over a long session. When explicitly enabled it now dedupes against what's
    already stored and stops at a cap.
    """
    if not settings.get("auto_memorize"):
        return
    if not post_text or len(post_text) < 40:
        return
    try:
        existing = store.get_memories(bot_id)
        auto = [m for m in existing if m["text"].startswith("I once said:")]
        if len(auto) >= config.FEED_MEMORY_CAP:
            return
        snippet = post_text.strip()
        for m in auto:
            if snippet[:60].lower() in m["text"].lower():
                return
        store.add_memory(bot_id, f'I once said: "{snippet}"', enabled=True)
    except Exception as e:
        print(f"[feed] auto-memorize failed for {bot_id}: {e}")


def purge_auto_memories(char_id):
    """Undo the damage the old always-on behaviour did to a character."""
    removed = 0
    for m in store.get_memories(char_id):
        if m["text"].startswith("I once said:"):
            if store.delete_memory(char_id, m["id"]):
                removed += 1
    return removed


# ─────────────────────────────────────────────────────────────
#  THE TURN
# ─────────────────────────────────────────────────────────────

def generate_bot_post():
    """think → remember → post → memorize. Runs on a background thread."""
    with feed_lock:
        if not feed_state["active_bots"]:
            feed_state["is_thinking"] = feed_state["is_posting"] = False
            return
        if feed_state["priority_queue"]:
            bot_id = feed_state["priority_queue"].pop(0)
        else:
            idx = feed_state["turn_index"] % len(feed_state["active_bots"])
            bot_id = feed_state["active_bots"][idx]
            feed_state["turn_index"] = (idx + 1) % len(feed_state["active_bots"])
        topic_injection = feed_state["topic_injection"]
        feed_state["topic_injection"] = None
        feed_state["current_thinker"] = bot_id
        feed_state["is_thinking"] = True

    try:
        bot = store.get_character(bot_id)
        if not bot:
            return
        bot_name = bot.get("name", "Unknown")
        feed_context = _recent_feed_text(n=settings.get("context_posts", 10))

        try:
            hits = store.RAG.search(feed_context[-200:], k=5, char_id=bot_id)
            memories = [h["text"] for h in hits if h["score"] > 0.15]
        except Exception:
            memories = []
        if not memories:
            memories = store.enabled_memories(bot_id)[:5]

        try:
            inner_thought = _run_think_step(bot, memories, feed_context, topic_injection)
        except Exception as e:
            print(f"[feed] think step failed for {bot_name}: {e}")
            inner_thought = "Let me consider this carefully."

        with feed_lock:
            feed_state["thoughts"][bot_id] = inner_thought
            feed_state["is_thinking"] = False
            feed_state["is_posting"] = True

        post_id = new_post_id()
        try:
            post_text, audio_urls = _run_post_step_streaming(
                bot, memories, feed_context, inner_thought, topic_injection, post_id)
        except Exception as e:
            print(f"[feed] post step failed for {bot_name}: {e}")
            post_text, audio_urls = "...", []

        post_text = post_text.strip()
        for mention in _detect_mentions(post_text):
            mid = _find_bot_by_name(mention)
            if mid and mid != bot_id:
                with feed_lock:
                    if mid not in feed_state["priority_queue"]:
                        feed_state["priority_queue"].append(mid)

        with feed_lock:
            feed_state["posts"].append(_blank_post(
                id=post_id, bot_id=bot_id, bot_name=bot_name, text=post_text,
                inner_thought=inner_thought, audio_files=audio_urls))
            if len(feed_state["posts"]) > config.FEED_MAX_POSTS:
                feed_state["posts"] = feed_state["posts"][-config.FEED_MAX_POSTS:]

        _auto_memorize(bot_id, post_text)
        save()
        gc_audio()

    except Exception as e:
        print(f"[feed] pipeline error: {e}")
    finally:
        with feed_lock:
            feed_state["is_thinking"] = feed_state["is_posting"] = False
            feed_state["current_thinker"] = None


def add_human_post(text, username="you"):
    for mention in _detect_mentions(text):
        mid = _find_bot_by_name(mention)
        if mid:
            with feed_lock:
                if mid not in feed_state["priority_queue"]:
                    feed_state["priority_queue"].append(mid)
    post = _blank_post(bot_name=username or "you", text=text, is_human=True)
    with feed_lock:
        feed_state["posts"].append(post)
    save()
    return post


def react(post_id, reaction):
    with feed_lock:
        for post in feed_state["posts"]:
            if post["id"] == post_id:
                post["reactions"][reaction] = post["reactions"].get(reaction, 0) + 1
                result = dict(post["reactions"])
                break
        else:
            return None
    save()
    return result


def clear():
    with feed_lock:
        feed_state["posts"] = []
        feed_state["thoughts"] = {}
        feed_state["priority_queue"] = []
        post_counter[0] = 0
    save()
    gc_audio()
    return True
