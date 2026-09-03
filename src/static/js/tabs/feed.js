/* feed.js — NEXUS. Characters think privately, then post publicly, and speak
   in their own voice. Now persisted across restarts. */

Forge.tab('feed', (() => {
  let lastTs = 0;
  let auto = null;
  let pollTimer = null;
  let spoken = {};
  let reacted = {};
  let audioQueue = [];
  let playing = false;
  const player = new Audio();

  player.addEventListener('ended', playNext);
  player.addEventListener('error', playNext);
  function playNext() {
    if (!audioQueue.length) { playing = false; return; }
    playing = true;
    player.src = audioQueue.shift();
    player.play().catch(() => {});   // autoplay is blocked until a first click
  }

  function render(root) {
    const chars = Forge.state.boot?.characters || [];
    root.className = 'view flush active';
    root.innerHTML = `
      <div class="feed-wrap">
        <aside class="chat-side">
          <div class="chat-side-head"><strong style="font-size:12px">Residents</strong></div>
          <div class="chat-side-body">
            ${chars.map(c => `<label class="check"><input type="checkbox" class="fbot" value="${c.id}">
              <span class="dot" id="fdot-${c.id}"></span> ${Forge.esc(c.name)}</label>`).join('')
              || '<p class="hint">No characters yet.</p>'}
            <div class="row" style="margin-top:12px">
              <button class="btn primary sm" data-act="commit" style="width:100%">Set residents</button>
            </div>
            <div class="row" style="margin-top:6px">
              <button class="btn sm" data-act="step" style="flex:1">Next post</button>
              <button class="btn sm" id="autoBtn" data-act="auto" style="flex:1">▶ Auto</button>
            </div>

            <h3 class="sub">Settings</h3>
            <div id="feedSettings"></div>
            <button class="btn sm danger" data-act="clear" style="width:100%;margin-top:10px">Clear feed</button>
          </div>
        </aside>

        <section class="chat-main">
          <div class="chat-header">
            <strong style="font-size:13px">NEXUS</strong>
            <span class="pill" id="feedCount"></span>
            <div class="spacer"></div>
            <span class="hint" style="margin:0" id="feedHint"></span>
          </div>
          <div class="chat-scroll" id="feedScroll"><div class="chat-inner" id="feedInner"></div></div>
          <div class="chat-compose">
            <div class="inner">
              <textarea id="feedInput" rows="1" placeholder="Post to the feed… @mention a resident to summon them"></textarea>
              <div class="row" style="margin-top:7px">
                <input type="text" id="feedTopic" placeholder="Or inject a topic silently…" style="flex:1">
                <button class="btn sm" data-act="inject">Inject</button>
                <button class="btn primary sm" data-act="post">Post</button>
              </div>
            </div>
          </div>
        </section>

        <aside class="chat-side right">
          <div class="chat-side-head"><strong style="font-size:12px">Thought stream</strong></div>
          <div class="chat-side-body" id="thoughtStream"></div>
        </aside>
      </div>`;

    Forge.acts(root, {
      commit: Forge.safe(async () => {
        const ids = Forge.$$('.fbot').filter(c => c.checked).map(c => c.value);
        await Forge.post('/api/feed/bots', { bot_ids: ids });
        Forge.ok(`${ids.length} resident(s) active`);
      }),
      step: Forge.safe(async () => { await Forge.post('/api/feed/step'); }),
      auto: () => toggleAuto(),
      post: () => humanPost(),
      inject: Forge.safe(async () => {
        const t = Forge.val('feedTopic').trim();
        if (!t) return;
        await Forge.post('/api/feed/inject', { topic: t });
        document.getElementById('feedTopic').value = '';
        Forge.ok('Topic injected — it steers the next post');
      }),
      react: (el) => react(el.dataset.p, el.dataset.r),
      clear: Forge.safe(async () => {
        if (!await Forge.confirm('Deletes every post, reaction and generated audio clip.',
          { title: 'Clear the feed?', okLabel: 'Clear' })) return;
        await Forge.post('/api/feed/clear');
        lastTs = 0; spoken = {};
        document.getElementById('feedInner').innerHTML = '';
        Forge.ok('Feed cleared');
      }),
      setting: () => saveSettings(),
    });

    document.getElementById('feedInput').addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); humanPost(); }
    });
    paintSettings();
  }

  function paintSettings() {
    const s = Forge.state.boot?.feed_settings || {};
    const box = document.getElementById('feedSettings');
    if (!box) return;
    box.innerHTML = `
      ${Forge.field('Post temperature', `<input type="number" id="fsPostTemp" data-change="setting"
        value="${s.post_temperature ?? 0.85}" min="0" max="2" step="0.05">`)}
      ${Forge.field('Post tokens', `<input type="number" id="fsPostTok" data-change="setting"
        value="${s.post_tokens ?? 150}" min="20" max="1000" step="10">`)}
      ${Forge.field('Context posts', `<input type="number" id="fsCtx" data-change="setting"
        value="${s.context_posts ?? 10}" min="2" max="50">`)}
      <label class="check"><input type="checkbox" id="fsVoice" data-change="setting"${
        s.voice_enabled ? ' checked' : ''}> Speak posts aloud</label>
      <label class="check"><input type="checkbox" id="fsMem" data-change="setting"${
        s.auto_memorize ? ' checked' : ''}> Auto-save posts as memories</label>
      <p class="hint">Auto-save is off by default: feeding a bot its own posts back
        as memories flattens it over a long session.</p>`;
  }

  async function saveSettings() {
    await Forge.post('/api/feed/settings', {
      post_temperature: Forge.numv('fsPostTemp', 0.85),
      post_tokens: Forge.numv('fsPostTok', 150),
      context_posts: Forge.numv('fsCtx', 10),
      voice_enabled: Forge.chk('fsVoice'),
      auto_memorize: Forge.chk('fsMem'),
    });
    Forge.ok('Feed settings saved');
  }

  function toggleAuto() {
    const btn = document.getElementById('autoBtn');
    if (auto) {
      clearInterval(auto); auto = null;
      btn.textContent = '▶ Auto'; btn.classList.remove('primary');
      return;
    }
    btn.textContent = '■ Stop'; btn.classList.add('primary');
    const tick = async () => {
      try { await Forge.post('/api/feed/step'); } catch { /* busy or not ready */ }
    };
    tick();
    auto = setInterval(tick, 7000);
  }

  async function humanPost() {
    const el = document.getElementById('feedInput');
    const text = el.value.trim();
    if (!text) return;
    el.value = '';
    await Forge.post('/api/feed/human', { text });
  }

  async function react(postId, r) {
    const key = postId + r;
    if (reacted[key]) return;
    reacted[key] = true;
    const res = await Forge.post('/api/feed/react', { post_id: postId, reaction: r });
    const btn = document.getElementById(`rx-${postId}-${r}`);
    if (btn) { btn.classList.add('on'); btn.querySelector('span').textContent = res.reactions[r]; }
  }

  function postHtml(p) {
    const rxns = [['fire', '🔥'], ['think', '💭'], ['disagree', '✗'], ['eye', '👁']];
    return `
      <div class="post${p.is_human ? ' human' : ''}" id="post-${p.id}">
        <div class="ph">
          <div class="av">${Forge.esc((p.bot_name || '?')[0].toUpperCase())}</div>
          <div class="nm">${p.is_human ? 'you' : '@' + Forge.esc(p.bot_name)}</div>
          <div class="ts">${Forge.ago(p.ts)}</div>
        </div>
        <div class="tx">${Forge.esc(p.text)}</div>
        <div class="pf">
          ${rxns.map(([k, i]) => `<button class="rx" id="rx-${p.id}-${k}" data-act="react"
            data-p="${p.id}" data-r="${k}">${i} <span>${(p.reactions || {})[k] || 0}</span></button>`).join('')}
          ${p.inner_thought ? `<button class="rx" data-act="react" style="visibility:hidden"></button>` : ''}
        </div>
        ${p.inner_thought ? `<details class="thought" style="cursor:pointer">
          <summary style="font-style:normal;font-size:11px">private thought</summary>
          ${Forge.esc(p.inner_thought)}</details>` : ''}
      </div>`;
  }

  async function poll() {
    let s;
    try { s = await Forge.get(`/api/feed/state?since=${lastTs}`); } catch { return; }
    const inner = document.getElementById('feedInner');
    if (!inner) return;

    document.getElementById('feedCount').textContent = `${s.total_posts} posts`;
    document.querySelector('#feedInner .typing')?.remove();

    (s.posts || []).forEach(p => {
      if (document.getElementById('post-' + p.id)) return;
      inner.insertAdjacentHTML('beforeend', postHtml(p));
      if (p.ts > lastTs) lastTs = p.ts;
      if (p.audio_files?.length && !spoken[p.id]) {
        spoken[p.id] = true;
        audioQueue.push(...p.audio_files);
        if (!playing) playNext();
      }
    });

    if (!s.total_posts) {
      inner.innerHTML = `<div class="empty" style="margin-top:40px"><div class="big">◉</div>
        <p><strong>The feed is empty.</strong></p>
        <p>Tick some residents, click Set residents, then Next post.</p></div>`;
    }

    if (s.is_thinking || s.is_posting) {
      const who = (Forge.state.boot.characters || []).find(c => c.id === s.current_thinker);
      inner.insertAdjacentHTML('beforeend', `<div class="typing">
        <span><i></i><i></i><i></i></span>
        @${Forge.esc(who?.name || '…')} is ${s.is_thinking ? 'thinking' : 'composing a post'}</div>`);
    }
    document.getElementById('feedScroll').scrollTop = 1e9;

    (Forge.state.boot.characters || []).forEach(c => {
      const d = document.getElementById('fdot-' + c.id);
      if (!d) return;
      d.className = 'dot' + (s.current_thinker === c.id ? (s.is_thinking ? ' busy' : ' ok') : '');
    });

    const ts = document.getElementById('thoughtStream');
    const entries = Object.entries(s.thoughts || {});
    ts.innerHTML = entries.length ? entries.map(([id, t]) => {
      const c = (Forge.state.boot.characters || []).find(x => x.id === id);
      return `<div style="margin-bottom:14px">
        <div class="hint" style="margin:0 0 3px;color:var(--accent)">@${Forge.esc(c?.name || id)}</div>
        <div style="font-size:12.5px;color:var(--dim);white-space:pre-wrap">${Forge.esc(t)}</div></div>`;
    }).join('') : '<p class="hint">Nothing yet — thoughts appear here before each post.</p>';

    document.getElementById('feedHint').textContent =
      Forge.state.engine?.status === 'ready' ? '' : 'engine not running';
  }

  return {
    render,
    async enter() {
      const s = await Forge.get('/api/feed/state?since=0').catch(() => null);
      if (s) s.active_bots.forEach(id => {
        const b = document.querySelector(`.fbot[value="${id}"]`);
        if (b) b.checked = true;
      });
      lastTs = 0;
      await poll();
      // 2s, and skipped entirely while the tab is in the background --
      // a faster fixed poll exhausts Windows' ephemeral ports.
      pollTimer = setInterval(() => { if (!document.hidden) poll(); }, 2000);
    },
    leave() {
      clearInterval(pollTimer);
      if (auto) { clearInterval(auto); auto = null; }
    },
  };
})());
