/* chat.js — the Inference tab.
   Chat list (new / rename / delete), streamed responses, and a settings panel
   whose values are saved per chat: system prompt, temperature, max tokens,
   tools, reasoning and memory injection. */

Forge.tab('chat', (() => {
  let chats = [];
  let chatId = null;
  let messages = [];
  let streaming = false;
  let abort = null;
  const chatPlayer = new Audio();
  let showSettings = true;

  const DEFAULTS = () => ({
    system_prompt: '',
    temperature: Forge.state.boot?.defaults?.chat?.temperature ?? 0.7,
    max_tokens: Forge.state.boot?.defaults?.chat?.max_tokens ?? 512,
    tools_enabled: false,
    use_memories: true,
    speak_replies: false,
    enable_thinking: false,
  });
  let settings = DEFAULTS();

  async function speak(text) {
    if (!text || !text.trim()) return;
    const status = document.getElementById('chatStatus');
    // A cloned voice cold-starts its worker process on first use -- roughly
    // 15-20s to load XTTS onto the GPU -- with nothing else in the UI to show
    // for it. Without this the tab just looks stuck.
    if (status) status.textContent = 'synthesizing speech… (can take ~15s on first use)';
    try {
      const r = await Forge.post('/api/chat/speak', { text, char_id: Forge.state.char });
      chatPlayer.src = r.url;
      try {
        await chatPlayer.play();
      } catch (playErr) {
        // Browsers block autoplay outside a direct user gesture, and enough
        // time has passed since Send was clicked that this can land outside
        // that window. Silently swallowing this used to mean the audio was
        // ready but never played and nothing said why -- surface a manual
        // play control instead.
        showManualPlay(r.url);
      }
    } catch (e) {
      Forge.err('Speech synthesis failed: ' + (e.message || e));
    } finally {
      if (status) status.textContent = '';
    }
  }

  function showManualPlay(url) {
    const under = document.querySelector('.chat-compose .under');
    if (!under) return;
    document.getElementById('manualPlayer')?.remove();
    const el = document.createElement('span');
    el.id = 'manualPlayer';
    el.className = 'row tight';
    el.innerHTML = `<span class="hint" style="margin:0">autoplay blocked —</span>
      <audio controls autoplay src="${url}" style="height:26px;vertical-align:middle"></audio>`;
    under.prepend(el);
  }

  function charName(id) {
    return (Forge.state.boot?.characters || []).find(c => c.id === id)?.name || '—';
  }

  // ── render ─────────────────────────────────────────────
  function render(root) {
    const chars = Forge.state.boot?.characters || [];
    if (!chars.length) {
      root.innerHTML = `<div style="flex:1;padding:40px"><div class="empty">
        <div class="big">☺</div>
        <p><strong>No characters yet.</strong></p>
        <p>Build one in the Persona tab, or create a plain one in Characters to
           chat with the base model through a system prompt.</p>
        <div class="row" style="justify-content:center;margin-top:14px">
          <button class="btn primary" data-act="mkchar">Create a character</button>
        </div></div></div>`;
      Forge.acts(root, { mkchar: () => Forge.go('characters') });
      return;
    }

    root.className = 'view flush active';
    root.innerHTML = `
      <div class="chat-wrap${showSettings ? '' : ' no-settings'}" id="chatWrap">

        <aside class="chat-side">
          <div class="chat-side-head">
            <select id="chatChar" style="flex:1">${chars.map(c =>
              `<option value="${c.id}"${c.id === Forge.state.char ? ' selected' : ''}>${Forge.esc(c.name)}</option>`
            ).join('')}</select>
            <button class="btn sm icon" data-act="new" title="New chat">＋</button>
          </div>
          <div class="chat-side-body" id="chatList"></div>
        </aside>

        <section class="chat-main">
          <div class="chat-header">
            <strong id="chatTitle" style="font-size:13px">New chat</strong>
            <span class="pill" id="chatMeta"></span>
            <div class="spacer"></div>
            <button class="btn sm" data-act="rename">Rename</button>
            <button class="btn sm" data-act="clear">Clear</button>
            <button class="btn sm" data-act="toggleSettings">${showSettings ? 'Hide' : 'Show'} settings</button>
          </div>
          <div class="chat-scroll" id="chatScroll"><div class="chat-inner" id="chatInner"></div></div>
          <div class="chat-compose">
            <div class="inner">
              <textarea id="chatInput" rows="1"
                placeholder="Message ${Forge.esc(charName(Forge.state.char))}…   (Enter to send, Shift+Enter for a newline)"></textarea>
              <div class="under">
                <span id="chatStatus"></span>
                <div class="spacer"></div>
                <button class="btn sm" id="stopBtn" data-act="stop" hidden>Stop</button>
                <button class="btn primary sm" data-act="send">Send</button>
              </div>
            </div>
          </div>
        </section>

        <aside class="chat-side right" id="settingsPane">
          <div class="chat-side-head"><strong style="font-size:12px">Chat settings</strong></div>
          <div class="chat-side-body" id="settingsBody"></div>
        </aside>
      </div>`;

    Forge.acts(root, {
      new: () => newChat(),
      send: () => send(),
      stop: () => { if (abort) abort.abort(); },
      rename: () => renameChat(),
      clear: () => clearChat(),
      toggleSettings: () => { showSettings = !showSettings; render(root); paintAll(); },
      open: (el) => openChat(el.dataset.id),
      del: (el) => deleteChat(el.dataset.id),
      resend: (el) => resend(+el.dataset.i),
      copy: (el) => {
        navigator.clipboard.writeText(messages[+el.dataset.i]?.content || '');
        Forge.ok('Copied');
      },
      applyPersona: () => {
        const c = (Forge.state.boot.characters || []).find(x => x.id === Forge.state.char);
        const el = document.getElementById('sysPrompt');
        el.value = c?.persona || '';
        settings.system_prompt = el.value;
        persist();
      },
      resetSettings: () => { settings = DEFAULTS(); paintSettings(); persist(); },
    });

    const input = document.getElementById('chatInput');
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
    });
    input.addEventListener('input', () => {
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 200) + 'px';
    });

    document.getElementById('chatChar').onchange = Forge.safe(async (e) => {
      Forge.state.char = e.target.value;
      chatId = null; messages = [];
      await loadChats();
    });
  }

  // ── settings pane ──────────────────────────────────────
  function paintSettings() {
    const body = document.getElementById('settingsBody');
    if (!body) return;
    body.innerHTML = `
      ${Forge.field('System prompt', `<textarea id="sysPrompt" class="mono" rows="7"
        placeholder="Leave empty to use this character's persona plus its memories.">${Forge.esc(settings.system_prompt)}</textarea>`,
        'overrides the persona')}
      <div class="row tight" style="margin:-6px 0 14px">
        <button class="btn sm" data-act="applyPersona">Load persona</button>
        <button class="btn sm" data-act="resetSettings">Reset</button>
      </div>

      ${Forge.field('Temperature', `<div class="row"><input type="range" id="tempRange"
          min="0" max="2" step="0.05" value="${settings.temperature}" style="flex:1">
        <span class="mono" id="tempVal" style="width:34px;text-align:right">${settings.temperature}</span></div>`,
        'higher = looser')}

      ${Forge.field('Max tokens', Forge.num('maxTok', settings.max_tokens, { min: 16, max: 8192, step: 16 }))}

      <label class="check"><input type="checkbox" id="useMem"${settings.use_memories ? ' checked' : ''}>
        Inject character memories</label>
      <label class="check"><input type="checkbox" id="useTools"${settings.tools_enabled ? ' checked' : ''}>
        Enable tools (web search, fetch, save memory)</label>
      <label class="check"><input type="checkbox" id="speakReplies"${settings.speak_replies ? ' checked' : ''}>
        Speak replies aloud</label>
      <label class="check"><input type="checkbox" id="enableThinking"${settings.enable_thinking ? ' checked' : ''}>
        Show reasoning</label>

      <p class="hint" style="margin-top:14px">
        These are saved with this chat, so each conversation keeps its own
        settings. Changing them mid-chat affects the next message only.</p>`;

    const t = document.getElementById('tempRange');
    t.oninput = () => { document.getElementById('tempVal').textContent = t.value; };
    t.onchange = () => { settings.temperature = parseFloat(t.value); persist(); };
    document.getElementById('maxTok').onchange = (e) => {
      settings.max_tokens = parseInt(e.target.value, 10) || 512; persist();
    };
    document.getElementById('sysPrompt').onchange = (e) => {
      settings.system_prompt = e.target.value; persist();
    };
    document.getElementById('useMem').onchange = (e) => {
      settings.use_memories = e.target.checked; persist();
    };
    document.getElementById('useTools').onchange = (e) => {
      settings.tools_enabled = e.target.checked; persist();
    };
    document.getElementById('speakReplies').onchange = (e) => {
      settings.speak_replies = e.target.checked; persist();
    };
    document.getElementById('enableThinking').onchange = (e) => {
      settings.enable_thinking = e.target.checked; persist();
    };
  }

  // ── chat list ──────────────────────────────────────────
  async function loadChats() {
    if (!Forge.state.char) return;
    chats = await Forge.get(`/api/characters/${Forge.state.char}/chats`);
    paintList();
    if (chats.length) await openChat(chatId && chats.some(c => c.id === chatId) ? chatId : chats[0].id);
    else await newChat();
  }

  function paintList() {
    const el = document.getElementById('chatList');
    if (!el) return;
    if (!chats.length) {
      el.innerHTML = `<p class="hint" style="padding:8px">No chats yet.</p>`;
      return;
    }
    el.innerHTML = chats.map(c => `
      <div class="chat-list-item${c.id === chatId ? ' active' : ''}" data-act="open" data-id="${c.id}">
        <span class="t">${Forge.esc(c.title || 'New chat')}</span>
        <span class="n">${c.messages}</span>
        <button class="x" data-act="del" data-id="${c.id}" title="Delete chat">✕</button>
      </div>`).join('');
  }

  async function newChat() {
    if (!Forge.state.char) return;
    const c = await Forge.post(`/api/characters/${Forge.state.char}/chats`,
      { title: 'New chat', settings: DEFAULTS() });
    chats.unshift({ ...c, messages: 0 });
    chatId = c.id;
    messages = [];
    settings = { ...DEFAULTS(), ...(c.settings || {}) };
    paintAll();
  }

  async function openChat(id) {
    const c = await Forge.get(`/api/characters/${Forge.state.char}/chats/${id}`);
    chatId = c.id;
    messages = c.messages || [];
    settings = { ...DEFAULTS(), ...(c.settings || {}) };
    paintAll();
  }

  async function deleteChat(id) {
    if (!await Forge.confirm('This deletes the conversation permanently.',
      { title: 'Delete chat?', okLabel: 'Delete' })) return;
    await Forge.del(`/api/characters/${Forge.state.char}/chats/${id}`);
    chats = chats.filter(c => c.id !== id);
    if (chatId === id) { chatId = null; messages = []; }
    if (chats.length) await openChat(chats[0].id); else await newChat();
    Forge.ok('Chat deleted');
  }

  async function renameChat() {
    const cur = chats.find(c => c.id === chatId);
    const box = Forge.modal(`
      <h3>Rename chat</h3>
      <label class="field" style="margin-top:12px">
        <input type="text" id="newTitle" value="${Forge.esc(cur?.title || '')}"></label>
      <div class="row" style="justify-content:flex-end">
        <button class="btn" data-x="c">Cancel</button>
        <button class="btn primary" data-x="s">Save</button></div>`);
    box.querySelector('[data-x=c]').onclick = Forge.closeModal;
    box.querySelector('[data-x=s]').onclick = Forge.safe(async () => {
      const title = box.querySelector('#newTitle').value.trim() || 'New chat';
      await persist(title);
      Forge.closeModal();
    });
    box.querySelector('#newTitle').focus();
  }

  async function clearChat() {
    if (!messages.length) return;
    if (!await Forge.confirm('Empties this conversation but keeps the chat and its settings.',
      { title: 'Clear messages?', okLabel: 'Clear' })) return;
    messages = [];
    await persist();
    paintMessages();
  }

  async function persist(title) {
    if (!chatId || !Forge.state.char) return;
    const saved = await Forge.put(`/api/characters/${Forge.state.char}/chats/${chatId}`,
      { messages, title, settings });
    const row = chats.find(c => c.id === chatId);
    if (row) { row.title = saved.title; row.messages = messages.length; }
    paintList();
    paintHeader();
  }

  // ── painting ───────────────────────────────────────────
  function paintAll() { paintList(); paintHeader(); paintMessages(); paintSettings(); }

  function paintHeader() {
    const t = document.getElementById('chatTitle');
    if (!t) return;
    const cur = chats.find(c => c.id === chatId);
    t.textContent = cur?.title || 'New chat';
    const meta = document.getElementById('chatMeta');
    meta.textContent = `${charName(Forge.state.char)} · temp ${settings.temperature}` +
      (settings.tools_enabled ? ' · tools' : '');
  }

  function paintMessages() {
    const inner = document.getElementById('chatInner');
    if (!inner) return;
    if (!messages.length) {
      inner.innerHTML = `<div class="empty" style="margin-top:40px">
        <div class="big">💬</div>
        <p><strong>Talking to ${Forge.esc(charName(Forge.state.char))}.</strong></p>
        <p>Their adapter has to be loaded in the running engine for the trained
           voice to come through — otherwise you are talking to the base model
           wearing a system prompt.</p></div>`;
      return;
    }
    inner.innerHTML = messages.map((m, i) => msgHtml(m, i)).join('');
    scrollDown();
  }

  function msgHtml(m, i) {
    const isUser = m.role === 'user';
    const who = isUser ? 'you' : charName(Forge.state.char);
    const hasThink = m.thinking != null;
    const thinkOpen = hasThink && !m._thinkClosed;
    const think = hasThink
      ? `<details class="think-wrap"${thinkOpen ? ' open' : ''}>
          <summary>${thinkOpen ? 'Thinking…' : 'Thought process'}</summary>
          <div class="think" id="think-${i}">${Forge.esc(m.thinking)}</div>
        </details>` : '';
    const tools = (m.tools || []).map(t =>
      `<div class="toolcall"><span class="n">${Forge.esc(t.name)}</span> ${
        t.status === 'done' ? '✓' : '…'}${
        t.args ? ' ' + Forge.esc(JSON.stringify(t.args)).slice(0, 120) : ''}</div>`).join('');
    const acts = isUser
      ? `<button data-act="resend" data-i="${i}">retry from here</button>`
      : `<button data-act="copy" data-i="${i}">copy</button>`;
    return `
      <div class="msg ${isUser ? 'user' : 'assistant'}">
        <div class="av">${isUser ? 'Y' : Forge.esc(who[0].toUpperCase())}</div>
        <div class="body">
          <div class="who">${Forge.esc(who)}<span class="acts">${acts}</span></div>
          ${think}${tools}
          <div class="text" id="msg-${i}">${Forge.esc(m.content)}</div>
        </div>
      </div>`;
  }

  function scrollDown() {
    const s = document.getElementById('chatScroll');
    if (s) s.scrollTop = s.scrollHeight;
  }

  // ── sending ────────────────────────────────────────────
  async function resend(i) {
    if (streaming) return;
    const msg = messages[i];
    if (!msg || msg.role !== 'user') return;
    messages = messages.slice(0, i);
    paintMessages();
    await dispatch(msg.content);
  }

  async function send() {
    if (streaming) return;
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    input.style.height = 'auto';
    await dispatch(text);
  }

  async function dispatch(text) {
    if (!Forge.state.engine || Forge.state.engine.status !== 'ready') {
      Forge.err('Engine not running — start it from the Engine tab.');
      return;
    }
    messages.push({ role: 'user', content: text });
    const reply = { role: 'assistant', content: '', tools: [], thinking: null };
    messages.push(reply);
    paintMessages();

    streaming = true;
    document.getElementById('stopBtn').hidden = false;
    document.getElementById('chatStatus').textContent = 'generating…';
    const idx = messages.length - 1;
    // Re-fetched after every full paintMessages() call below (a tool event or
    // a thinking-block transition both trigger one) -- paintMessages()
    // replaces the whole message list's innerHTML, so the old node this
    // pointed at is detached and silently stops updating otherwise.
    let target = document.getElementById('msg-' + idx);
    const refreshTarget = () => { target = document.getElementById('msg-' + idx); };
    if (target) target.classList.add('caret');

    abort = new AbortController();
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abort.signal,
        body: JSON.stringify({
          char_id: Forge.state.char,
          messages: messages.slice(0, -1).map(m => ({ role: m.role, content: m.content })),
          system_prompt: settings.system_prompt,
          temperature: settings.temperature,
          max_tokens: settings.max_tokens,
          tools_enabled: settings.tools_enabled,
          use_memories: settings.use_memories,
          enable_thinking: settings.enable_thinking,
        }),
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.error || `${res.status} ${res.statusText}`);
      }

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        // SSE frames are separated by a blank line; keep the trailing partial.
        const frames = buf.split('\n\n');
        buf = frames.pop();
        for (const frame of frames) {
          const line = frame.split('\n').find(l => l.startsWith('data:'));
          if (!line) continue;
          let ev;
          try { ev = JSON.parse(line.slice(5).trim()); } catch { continue; }
          if (ev.type === 'thinking') {
            if (reply.thinking === null) {
              // First reasoning token: the bubble doesn't exist in the DOM
              // yet (msgHtml() renders nothing when m.thinking is still
              // null), so this one has to be a full repaint. Every
              // subsequent thinking delta updates the node directly instead.
              reply.thinking = '';
              paintMessages();
              refreshTarget();
            }
            reply.thinking += ev.text;
            const tEl = document.getElementById('think-' + idx);
            if (tEl) tEl.textContent = reply.thinking;
            scrollDown();
          } else if (ev.type === 'token') {
            if (reply.thinking != null && !reply._thinkClosed) {
              // First real reply token after a reasoning block: collapse it,
              // same as the other AI front ends this was modeled on.
              reply._thinkClosed = true;
              paintMessages();
              refreshTarget();
            }
            reply.content += ev.text;
            if (target) target.textContent = reply.content;
            scrollDown();
          } else if (ev.type === 'tool') {
            const found = reply.tools.find(t => t.name === ev.name && t.status !== 'done');
            if (found) Object.assign(found, ev);
            else reply.tools.push({ name: ev.name, args: ev.args, status: ev.status });
            paintMessages();
            refreshTarget();
          } else if (ev.type === 'error') {
            throw new Error(ev.error);
          }
        }
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        reply.content += '\n\n(stopped)';
      } else {
        Forge.err(e);
        reply.content = reply.content || `(failed: ${e.message})`;
      }
    } finally {
      streaming = false;
      abort = null;
      document.getElementById('stopBtn').hidden = true;
      document.getElementById('chatStatus').textContent = '';
      if (target) target.classList.remove('caret');
      // Safety net for the edge case where the stream ends (error, abort, or
      // hit max_tokens) while still inside the reasoning block -- without
      // this the bubble would render "Thinking…" forever after the fact.
      if (reply.thinking != null) reply._thinkClosed = true;
      paintMessages();

      if (settings.speak_replies && reply.content && !reply.content.startsWith('(failed')) {
        speak(reply.content.replace(/\(stopped\)$/, '').trim());
      }

      // Name the chat off the first exchange so the sidebar isn't ten rows
      // of "New chat".
      const cur = chats.find(c => c.id === chatId);
      let title;
      if (cur && (!cur.title || cur.title === 'New chat') && messages.length >= 2) {
        try {
          title = (await Forge.post('/api/chat/title', { text })).title;
        } catch { title = text.slice(0, 40); }
      }
      await persist(title);
    }
  }

  return {
    render,
    async enter() { await loadChats(); paintSettings(); },
    leave() { if (abort) abort.abort(); chatPlayer.pause(); },
  };
})());
