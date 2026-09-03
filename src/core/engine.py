import os, json, time, threading, urllib.request, urllib.error, subprocess

# ── Tool-call wire protocol ─────────────────────────────────
# To use a tool the model must emit EXACTLY one block and nothing else:
#   [[TOOL]]{"name":"web_search","args":{"query":"..."}}[[/TOOL]]
TOOL_OPEN  = "[[TOOL]]"
TOOL_CLOSE = "[[/TOOL]]"
MAX_TOOL_HOPS = 4


def _split_think(stream):
    """Splits a raw delta stream into ('think', text) / ('text', text) pairs,
    stripping a leading <think>...</think> block (Qwen3's reasoning span) out
    of the visible reply. Only recognizes it right at the very start of the
    response -- a reply that doesn't open with <think> is passed through
    completely untouched, so this can never mangle normal prose that happens
    to mention the word.

    Single running buffer, resolved incrementally as each delta arrives, so
    a tag split across two network chunks (e.g. '</th' + 'ink>') still gets
    caught cleanly -- the same shape of problem chat_stream's tool-call
    detection already solves, just for a different boundary marker."""
    OPEN, CLOSE = "<think>", "</think>"
    raw = ""
    resolved_open = False   # committed: this response does have a think block
    closed = False          # already found </think> (or ruled a block out)
    for delta in stream:
        if delta is None:
            continue
        raw += delta
        while raw:
            if closed:
                yield ("text", raw)
                raw = ""
                break
            if not resolved_open:
                probe = raw.lstrip()
                if probe.startswith(OPEN):
                    raw = probe[len(OPEN):]
                    resolved_open = True
                    continue
                if len(probe) < len(OPEN) and OPEN.startswith(probe):
                    break   # could still become "<think>" — wait for more
                closed = True
                yield ("text", raw)
                raw = ""
                break
            # inside an opened think block, looking for the close tag
            if CLOSE in raw:
                head, raw = raw.split(CLOSE, 1)
                if head:
                    yield ("think", head)
                closed = True
                continue
            # No close tag yet — yield what's safe, holding back enough of
            # the tail that a split CLOSE marker can't slip through as text.
            hold = len(CLOSE) - 1
            if len(raw) > hold:
                yield ("think", raw[:-hold])
                raw = raw[-hold:]
            break
    if raw:
        yield ("think" if resolved_open and not closed else "text", raw)


def find_llama_server_bin(hint=""):
    names = ["llama-server.exe", "llama-server", "server.exe", "server"]
    subdirs = ["", os.path.join("build", "bin"), os.path.join("build", "Release"),
               os.path.join("build", "Debug"), "bin"]
    roots = []
    if hint:
        roots.append(hint)
    import config as _cfg
    roots += [_cfg.LLAMA_CPP_DIR,            # runtime/llama.cpp -- the real one
              ".", "llama.cpp", "../llama.cpp", os.path.expanduser("~/llama.cpp"),
              r"C:\llama.cpp", r"C:\tools\llama.cpp"]
    for r in roots:
        for sd in subdirs:
            for n in names:
                p = os.path.join(r, sd, n) if sd else os.path.join(r, n)
                if os.path.exists(p):
                    return os.path.abspath(p)
    return None

class LlamaManager:
    """Owns a single llama-server process and the list of LoRA adapters it loaded."""

    def __init__(self):
        self.proc = None
        self.port = 8088
        self.ctx = 4096
        self.parallel = 1         # number of server slots; >1 isolates chat vs background
        self.base_gguf = None
        self.adapters = []        # list of {"name": char_id, "path": gguf_path, "id": idx}
        self.log = []
        self.status = "stopped"   # stopped | starting | ready | error
        self.error = None
        self._lock = threading.Lock()

    # ── lifecycle ──────────────────────────────────────────
    def _glog(self, m):
        print(m)
        self.log.append(m)
        if len(self.log) > 500:
            del self.log[:-500]

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def is_ready(self):
        return self.status == "ready" and self.proc and self.proc.poll() is None

    def health_probe(self, timeout=3):
        """Returns True iff the server process is alive AND the HTTP /health endpoint
        responds with 200.  More reliable than is_ready() in the window immediately
        after a server crash where the OS hasn't yet reaped the process (proc.poll()
        still returns None) but HTTP connections are already being refused or reset."""
        if not (self.proc and self.proc.poll() is None):
            self.status = "stopped"
            return False
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=timeout) as r:
                return r.status == 200
        except Exception:
            return False
        
    def adapter_id(self, char_id):
        for a in self.adapters:
            if a["name"] == char_id:
                return a["id"]
        return None

    def state(self):
        return {
            "status": self.status,
            "error": self.error,
            "base_gguf": self.base_gguf,
            "port": self.port,
            "ctx": self.ctx,
            "parallel": self.parallel,
            "adapters": [{"name": a["name"], "id": a["id"]} for a in self.adapters],
            "log": self.log[-120:],
        }

    def stop(self):
        with self._lock:
            if self.proc and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=10)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                    try:
                        self.proc.wait(timeout=5)
                    except Exception:
                        pass
            self.proc = None
            self.status = "stopped"
            self.adapters = []
            self.base_gguf = None
            self._glog("⏹ llama-server stopped")

    def start(self, base_gguf, adapters, llama_hint="", ctx=4096, ngl=999, port=8088,
              parallel=1):

        self.stop()
        with self._lock:
            self.status = "starting"
            self.error = None
            self.log = []
            self.port = port
            self.ctx = ctx
            self.parallel = max(1, int(parallel))
            self.base_gguf = base_gguf

            bin_path = find_llama_server_bin(llama_hint)
            if not bin_path:
                self.status = "error"
                self.error = ("llama-server binary not found. Build llama.cpp or set the "
                              "llama.cpp path. Looked for llama-server(.exe).")
                self._glog("✗ " + self.error)
                return False
            if not os.path.exists(base_gguf):
                self.status = "error"
                self.error = f"Base GGUF not found: {base_gguf}"
                self._glog("✗ " + self.error)
                return False

            total_ctx = ctx * self.parallel   # each slot keeps the full ctx
            cmd = [bin_path, "-m", base_gguf, "-c", str(total_ctx), "-ngl", str(ngl),
                   "--host", "127.0.0.1", "--port", str(port),
                   "--lora-init-without-apply"]
            if self.parallel > 1:
                cmd += ["--parallel", str(self.parallel), "--cont-batching"]
            self.adapters = []
            idx = 0
            for char_id, apath in adapters:
                if apath and os.path.exists(apath):
                    cmd += ["--lora", apath]
                    self.adapters.append({"name": char_id, "path": apath, "id": idx})
                    idx += 1
                else:
                    self._glog(f"⚠ adapter for {char_id} missing, skipping: {apath}")

            self._glog("▶ " + " ".join(cmd))
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True)
            except Exception as e:
                self.status = "error"
                self.error = f"Failed to launch llama-server: {e}"
                self._glog("✗ " + self.error)
                return False

        threading.Thread(target=self._drain_logs, daemon=True).start()
        threading.Thread(target=self._wait_ready, daemon=True).start()
        return True

    def _drain_logs(self):
        p = self.proc
        if not p or not p.stdout:
            return
        for line in p.stdout:
            self._glog(line.rstrip())

    def _wait_ready(self, timeout=180):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not self.proc or self.proc.poll() is not None:
                self.status = "error"
                self.error = "llama-server exited during startup (see log)."
                return
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=2) as r:
                    if r.status == 200:
                        self.status = "ready"
                        self._glog(f"✓ llama-server ready on {self.base_url}")
                        return
            except Exception:
                pass
            time.sleep(1)
        self.status = "error"
        self.error = "Timed out waiting for llama-server /health."

    # ── tools ───────────────────────────────────────────────
    def _run_tool(self, name, args, char_id):
        if name == "web_search":
            return _web_search(args.get("query", ""))
        if name == "fetch_url":
            return _fetch_url(args.get("url", ""))
        if name == "save_memory":
            import store
            content = (args.get("content") or "").strip()
            if not content:
                return "No content given to save."
            store.add_memory(char_id, content, enabled=True)
            return f"Saved to memory: {content}"
        return f"Unknown tool '{name}'."

    # ── chat (streaming + agent loop) ───────────────────────
    def chat_stream(self, messages, char_id, temperature=0.7, max_tokens=512,
                    tools_enabled=True, enable_thinking=False):
        """Yields SSE-ready dict events: {type: thinking|token|tool|done|error, ...}.

        enable_thinking mirrors every other call site in this codebase
        (feed's think/post steps, eval, title generation) in defaulting OFF —
        interactive chat should be fast by default. Turn it on per-chat from
        the Inference tab's settings to see Qwen3's reasoning span; when it's
        on, a leading <think>...</think> block is split out into its own
        'thinking' events instead of leaking into the reply as visible text,
        which is what happened before this existed.
        """
        if not self.is_ready():
            yield {"type": "error", "error": "Engine not running. Start it first."}
            return

        lora_id = self.adapter_id(char_id)
        # Activate only this character's adapter (others to scale 0). None => base only.
        lora_field = [{"id": a["id"], "scale": 1.0 if a["id"] == lora_id else 0.0}
                      for a in self.adapters]

        convo = list(messages)
        hops = 0
        while True:
            hops += 1
            payload = {
                "messages": convo,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                "cache_prompt": True,   # reuse the slot's KV cache across turns → fast
                "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
            }
            if self.parallel > 1:
                payload["id_slot"] = 0   # user chat owns slot 0
            if lora_field:
                payload["lora"] = lora_field

            try:
                raw_stream = self._post_stream("/v1/chat/completions", payload)
            except Exception as e:
                yield {"type": "error", "error": f"llama-server request failed: {e}"}
                return

            TOOL_DECISION_CHARS = 60
            buf, decided, is_tool, toolbuf = "", False, False, ""
            for kind, delta in _split_think(raw_stream):
                if kind == "think":
                    yield {"type": "thinking", "text": delta}
                    continue
                if decided and not is_tool:
                    yield {"type": "token", "text": delta}
                    continue
                if decided and is_tool:
                    toolbuf += delta
                    continue

                buf += delta
                stripped = buf.lstrip()
                if not stripped:
                    continue

                # Tool call detected anywhere in the buffer
                if TOOL_OPEN in stripped:
                    # Trim any preamble before [[TOOL]]
                    idx = stripped.index(TOOL_OPEN)
                    toolbuf = stripped[idx:]
                    decided = True
                    is_tool = True
                    continue

                # Buffer is growing but still might start a tool call — keep buffering
                # as long as we haven't ruled it out and haven't buffered too much.
                still_possible = any(
                    TOOL_OPEN.startswith(stripped[-i:]) for i in range(1, len(TOOL_OPEN) + 1)
                    if len(stripped) >= i
                )
                if not decided and len(stripped) < TOOL_DECISION_CHARS and still_possible:
                    continue

                # Definitely a normal response — flush buffer and stream the rest
                decided = True
                yield {"type": "token", "text": buf}

            if not is_tool:
                # plain answer hop already streamed
                yield {"type": "done"}
                return

            # ── handle tool call ───────────────────────────
            if not tools_enabled or hops > MAX_TOOL_HOPS:
                # fall back: just show whatever it produced
                yield {"type": "token", "text": toolbuf}
                yield {"type": "done"}
                return

            raw = toolbuf
            inner = raw
            if TOOL_OPEN in inner:
                inner = inner.split(TOOL_OPEN, 1)[1]
            if TOOL_CLOSE in inner:
                inner = inner.split(TOOL_CLOSE, 1)[0]
            try:
                call = json.loads(inner.strip())
                name = call.get("name", "")
                args = call.get("args", {}) or {}
            except Exception:
                yield {"type": "token", "text": "\n\n(could not parse tool call)\n"}
                yield {"type": "done"}
                return

            yield {"type": "tool", "name": name, "args": args, "status": "running"}
            result = self._run_tool(name, args, char_id)
            yield {"type": "tool", "name": name, "status": "done", "result": result[:4000]}

            convo.append({"role": "assistant", "content": raw.strip()})
            convo.append({"role": "user",
                          "content": f"[tool_result name={name}]\n{result}\n[/tool_result]\n"
                                     f"Use this result to answer my previous message. "
                                     f"Do not call the same tool again unless truly needed."})
            # loop for the next hop (which streams the final answer)

    def _completion_payload(self, messages, temperature, max_tokens, char_id, enable_thinking):
        lora_id = self.adapter_id(char_id) if char_id else None
        lora_field = [{"id": a["id"], "scale": 1.0 if a["id"] == lora_id else 0.0}
                      for a in self.adapters]
        payload = {"messages": list(messages), "temperature": temperature,
                   "max_tokens": max_tokens, "stream": True, "cache_prompt": True}
        if self.parallel > 1:
            payload["id_slot"] = 1   # background cognition stays off the chat slot
        if lora_field:
            payload["lora"] = lora_field
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
        return payload

    def complete_stream(self, messages, char_id=None, temperature=0.8, max_tokens=300,
                        enable_thinking=None):
        """Streaming completion — yields text deltas as the model generates them.
        Same activation rules as complete(). Lets a caller start doing something
        (e.g. per-sentence TTS) before the full response is done."""
        if not self.is_ready():
            raise RuntimeError("Engine not running.")
        payload = self._completion_payload(messages, temperature, max_tokens, char_id, enable_thinking)
        for delta in self._post_stream("/v1/chat/completions", payload):
            if delta:
                yield delta

    def complete(self, messages, char_id=None, temperature=0.8, max_tokens=300,
                enable_thinking=None):
        """Non-streaming completion. Activates char_id's LoRA if given (else base only).
        Used by the mind loop so each inner voice can speak through its own region.
        enable_thinking=False skips Qwen3's <think> reasoning span (faster, and
        required for building non-thinking persona training data)."""
        return "".join(self.complete_stream(messages, char_id, temperature, max_tokens,
                                            enable_thinking)).strip()

    def _post_stream(self, path, payload):
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=600)
        try:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        yield delta
                except Exception:
                    continue
        finally:
            try:
                resp.close()
            except Exception:
                pass


# ── Weather keywords — any query containing these gets a live weather lookup ─
_WEATHER_WORDS = frozenset([
    "weather", "temperature", "forecast", "rain", "snow", "wind", "humidity",
    "sunny", "cloudy", "storm", "thunder", "celsius", "fahrenheit", "degrees",
    "hot", "cold", "warm", "chilly", "freezing", "fog", "foggy", "hail", "sleet",
    "overcast", "drizzle", "conditions", "climate",
])

# WMO weather interpretation codes (used by Open-Meteo)
_WMO_CODES = {
    0:"Clear sky", 1:"Mainly clear", 2:"Partly cloudy", 3:"Overcast",
    45:"Fog", 48:"Icy fog",
    51:"Light drizzle", 53:"Moderate drizzle", 55:"Dense drizzle",
    61:"Slight rain", 63:"Moderate rain", 65:"Heavy rain",
    71:"Slight snow", 73:"Moderate snow", 75:"Heavy snow", 77:"Snow grains",
    80:"Slight showers", 81:"Moderate showers", 82:"Heavy showers",
    85:"Slight snow showers", 86:"Heavy snow showers",
    95:"Thunderstorm", 96:"Thunderstorm with hail", 99:"Thunderstorm with heavy hail",
}


def _extract_location(query):
    """Strip generic weather/query words to isolate the place name."""
    strip_words = frozenset([
        "current", "today", "now", "what", "whats", "wahst", "waht", "wats",
        "is", "the", "in", "for", "at", "about", "a",
        "weather", "temperature", "temp", "temps", "forecast", "conditions",
        "outside", "like", "how", "check", "get", "find", "search",
        "tell", "me", "show", "right", "rn", "atm", "currently", "please",
    ])
    import re as _re
    tokens = _re.sub(r"[?.,!']", "", query.lower()).split()
    loc_tokens = [t for t in tokens if t not in strip_words]
    return " ".join(loc_tokens).strip() or query


def _location_candidates(query):
    """
    Return a list of location strings to try in order, most-specific first.
    Handles leading typos/noise: 'wahst iowa' → tries 'wahst iowa', then 'iowa'.
    """
    base = _extract_location(query)
    words = base.split()
    seen, candidates = set(), []
    # Progressively drop leading words — so noise/typos at the front get peeled off
    for start in range(min(3, len(words))):
        loc = " ".join(words[start:]).strip()
        if loc and loc not in seen:
            candidates.append(loc)
            seen.add(loc)
    return candidates or [query]


def _openmeteo_weather(location):

    try:
        import urllib.request, urllib.parse, json as _json
        # Step 1: geocode location name → lat/lon
        geo_url = (f"https://geocoding-api.open-meteo.com/v1/search"
                   f"?name={urllib.parse.quote(location)}&count=1&language=en&format=json")
        with urllib.request.urlopen(geo_url, timeout=6) as r:
            geo = _json.loads(r.read())
        if not geo.get("results"):
            return None
        res   = geo["results"][0]
        lat   = res["latitude"]
        lon   = res["longitude"]
        place = ", ".join(p for p in [res.get("name",""), res.get("admin1",""),
                                      res.get("country","")] if p)
        # Step 2: fetch current conditions
        wx_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
            f"weather_code,wind_speed_10m"
            f"&temperature_unit=fahrenheit&wind_speed_unit=mph&forecast_days=1"
        )
        with urllib.request.urlopen(wx_url, timeout=6) as r:
            wx = _json.loads(r.read())
        cur   = wx["current"]
        code  = int(cur.get("weather_code", 0))
        desc  = _WMO_CODES.get(code, f"Code {code}")
        temp  = cur.get("temperature_2m",      "?")
        feels = cur.get("apparent_temperature", "?")
        humid = cur.get("relative_humidity_2m", "?")
        wind  = cur.get("wind_speed_10m",       "?")
        return (f"{place}: {desc}, {temp}°F (feels like {feels}°F), "
                f"humidity {humid}%, wind {wind}mph")
    except Exception:
        return None


def _wttr_weather(location):

    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    # Try structured JSON first
    try:
        import urllib.request, urllib.parse, json as _json
        loc = urllib.parse.quote_plus(location)
        url = f"https://wttr.in/{loc}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = _json.loads(r.read())
        cc     = data["current_condition"][0]
        area   = data.get("nearest_area", [{}])[0]
        city   = area.get("areaName",  [{}])[0].get("value", "")
        region = area.get("region",    [{}])[0].get("value", "")
        desc   = cc.get("weatherDesc", [{}])[0].get("value", "Unknown")
        temp_f = cc.get("temp_F",      "?")
        feel_f = cc.get("FeelsLikeF",  "?")
        humid  = cc.get("humidity",    "?")
        wind_m = cc.get("windspeedMiles", "?")
        wind_d = cc.get("winddir16Point", "")
        place  = ", ".join(p for p in [city, region] if p) or location
        return (f"{place}: {desc}, {temp_f}°F (feels like {feel_f}°F), "
                f"humidity {humid}%, wind {wind_m}mph {wind_d}").strip()
    except Exception:
        pass
    # Fallback: simple one-line text format
    try:
        import urllib.request, urllib.parse
        loc = urllib.parse.quote_plus(location)
        url = f"https://wttr.in/{loc}?format=3"
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=6) as r:
            result = r.read().decode("utf-8", "ignore").strip()
        if result and "Unknown location" not in result and len(result) > 3:
            return result
    except Exception:
        pass
    return None


def _fetch_url(url):
    url = (url or "").strip()
    if not url:
        return "No URL provided."
    if not url.startswith(("http://", "https://")):
        return f"Invalid URL (must start with http:// or https://): {url}"
    try:
        import urllib.request, re as _re
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text" not in content_type and "json" not in content_type:
                return f"URL returned non-text content ({content_type}), cannot read."
            raw = resp.read(200_000).decode("utf-8", "ignore")

        # Strip scripts, styles, and HTML tags
        raw = _re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=_re.DOTALL | _re.IGNORECASE)
        raw = _re.sub(r"<style[^>]*>.*?</style>",  " ", raw, flags=_re.DOTALL | _re.IGNORECASE)
        raw = _re.sub(r"<[^>]+>", " ", raw)
        raw = _re.sub(r"&[a-zA-Z]+;", " ", raw)   # HTML entities
        raw = _re.sub(r"[ \t]{2,}", " ", raw)
        raw = _re.sub(r"\n{3,}", "\n\n", raw.strip())

        text = raw[:5000].strip()
        return f"Content from {url}:\n\n{text}" + ("\n\n[truncated]" if len(raw) > 3000 else "")
    except Exception as e:
        return f"fetch_url failed for {url}: {e}"


def _web_search(query):
    query = (query or "").strip()
    if not query:
        return "Empty search query."

    q_lower = query.lower()
    is_weather = any(w in q_lower for w in _WEATHER_WORDS)

    # ── Live weather data (Open-Meteo primary, wttr.in backup) ───────────────
    weather_prefix = ""
    if is_weather:
        live = None
        for loc in _location_candidates(query):
            live = _openmeteo_weather(loc) or _wttr_weather(loc)
            if live:
                break
        if live:
            weather_prefix = f"Live weather data:\n  {live}\n\n"
        else:
            # Be explicit so the model doesn't hallucinate a received result
            weather_prefix = (
                "Live weather lookup failed (could not reach weather APIs or "
                "could not identify the location). "
                "Do not claim to have received live weather data. "
                "Use the web results below if helpful, or tell the user you "
                "could not get current conditions.\n\n"
            )

    # ── General web search (ddgs / duckduckgo_search) ─────────────────────────
    DDGS_cls = None
    for _mod in ("ddgs", "duckduckgo_search"):
        try:
            import importlib
            _m = importlib.import_module(_mod)
            DDGS_cls = getattr(_m, "DDGS", None)
            if DDGS_cls:
                break
        except Exception:
            continue

    search_results = ""
    if DDGS_cls is not None:
        try:
            try:
                with DDGS_cls() as ddg:
                    results = list(ddg.text(query, max_results=5))
            except TypeError:
                results = list(DDGS_cls().text(query, max_results=5))
            out = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                body  = r.get("body", "")
                href  = r.get("href", "")
                out.append(f"{i}. {title}\n   {body}\n   {href}")
            if out:
                search_results = "Web search results:\n" + "\n".join(out)
        except Exception:
            pass

    # ── DuckDuckGo instant-answers fallback (no package) ─────────────────────
    if not search_results:
        try:
            import urllib.request, urllib.parse, json as _json
            encoded = urllib.parse.quote_plus(query)
            url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
            req = urllib.request.Request(url, headers={"User-Agent": "LoraForge-Mind/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
            out = []
            abstract = data.get("AbstractText", "").strip()
            if abstract:
                out.append(f"1. {data.get('Heading','')}\n   {abstract}\n   {data.get('AbstractURL','')}")
            for topic in data.get("RelatedTopics", [])[:4]:
                if isinstance(topic, dict) and topic.get("Text"):
                    out.append(f"{len(out)+1}. {topic['Text']}\n   {topic.get('FirstURL','')}")
            if out:
                note = " (install `ddgs` for richer results: pip install ddgs)"
                search_results = f"Search results{note}:\n" + "\n".join(out)
        except Exception:
            pass

    # ── Assemble final result ─────────────────────────────────────────────────
    combined = weather_prefix + search_results

    if combined.strip():
        return combined.strip()

    if is_weather and not weather_prefix:
        return ("Could not fetch live weather. "
                "Try: pip install ddgs  for web search results.")
    if not DDGS_cls:
        return ("web_search unavailable — ddgs not installed in this environment. "
                "Activate your lora_env and run: pip install ddgs")
    return f"No results found for '{query}'."


def build_system_prompt(persona, memories, tools_enabled):
    import datetime
    parts = []

    # Always stamp the current server date/time so the model never hallucinates
    # or wastes a tool call searching for it.
    now_utc   = datetime.datetime.now(datetime.timezone.utc)
    now_local = datetime.datetime.now()
    parts.append(
        f"Current date/time: {now_utc.strftime('%A, %B %d, %Y')} — "
        f"{now_utc.strftime('%H:%M')} UTC  /  "
        f"{now_local.strftime('%H:%M')} server local time. "
        f"Use this for any questions about today's date or current time — "
        f"do NOT call a tool for this."
    )

    if persona.strip():
        parts.append(persona.strip())
    if memories:
        parts.append("Things you remember (treat as known facts):\n" +
                     "\n".join(f"- {m}" for m in memories))
    if tools_enabled:
        parts.append(
            "TOOL USE — READ THIS CAREFULLY:\n"
            "You have live tools. You MUST call web_search — not answer from memory — whenever "
            "the user asks about anything real-time or current, including:\n"
            "  • weather (current, today, forecast, temperature)\n"
            "  • news, recent events, anything that happened recently\n"
            "  • prices, stock values, exchange rates\n"
            "  • sports scores, standings, schedules\n"
            "  • any fact that could have changed since your training\n\n"
            "Do NOT use web_search for the current date or time — that is already in your system context above.\n\n"
            "Your training data is outdated for everything else. Call the tool FIRST, then answer.\n\n"
            "If web_search returns links but not the information you need, call fetch_url "
            "with one of those links to read the actual page content.\n\n"
            "To call a tool, output ONLY this block with ZERO text before it:\n"
            f"{TOOL_OPEN}{{\"name\":\"tool_name\",\"args\":{{\"key\":\"value\"}}}}{TOOL_CLOSE}\n\n"
            "Available tools:\n"
            f"  web_search  — {TOOL_OPEN}{{\"name\":\"web_search\",\"args\":{{\"query\":\"your search\"}}}}{TOOL_CLOSE}\n"
            f"  fetch_url   — {TOOL_OPEN}{{\"name\":\"fetch_url\",\"args\":{{\"url\":\"https://...\"}}}}{TOOL_CLOSE}\n"
            f"  save_memory — {TOOL_OPEN}{{\"name\":\"save_memory\",\"args\":{{\"content\":\"fact to remember\"}}}}{TOOL_CLOSE}\n\n"
            "After the tool result is shown to you, answer the user in plain text. "
            "Do not call a tool a second time unless the result was empty or irrelevant."
        )
    return "\n\n".join(parts) if parts else "You are a helpful assistant."


# Module-level singleton used by app.py
MANAGER = LlamaManager()