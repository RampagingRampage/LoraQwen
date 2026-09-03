/* app.js — shell, router, shared state and the small helpers every tab uses.
   Tab modules register themselves with Forge.tab(name, {render, enter, leave}). */

const Forge = (() => {
  const tabs = {};
  const state = {
    boot: null,
    current: null,
    char: null,          // selected character id, shared across tabs
    dataset: null,       // selected dataset name, shared across tabs
    persona: null,
    poll: null,
  };

  // ── API ────────────────────────────────────────────────
  async function api(path, opts = {}) {
    const o = { headers: {}, ...opts };
    if (o.body && !(o.body instanceof FormData)) {
      o.headers['Content-Type'] = 'application/json';
      o.body = JSON.stringify(o.body);
    }
    const res = await fetch(path, o);
    const ct = res.headers.get('content-type') || '';
    const data = ct.includes('json') ? await res.json().catch(() => ({})) : await res.text();
    if (!res.ok) throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
    return data;
  }
  const get  = (p) => api(p);
  const post = (p, body) => api(p, { method: 'POST', body });
  const put  = (p, body) => api(p, { method: 'PUT', body });
  const del  = (p, body) => api(p, { method: 'DELETE', body });

  // ── toasts ─────────────────────────────────────────────
  function toast(msg, kind = '') {
    const el = document.createElement('div');
    el.className = 'toast ' + kind;
    el.textContent = msg;
    document.getElementById('toasts').appendChild(el);
    setTimeout(() => el.remove(), kind === 'err' ? 7000 : 3600);
  }
  const ok  = (m) => toast(m, 'ok');
  const err = (m) => toast(typeof m === 'string' ? m : (m && m.message) || 'Something went wrong', 'err');

  // Wrap an async handler so a rejected promise surfaces as a toast rather
  // than a silent console error the user never sees.
  const safe = (fn) => (...a) => Promise.resolve(fn(...a)).catch(err);

  // ── modal ──────────────────────────────────────────────
  function modal(html, { wide = false } = {}) {
    const bg = document.getElementById('modalBg');
    const box = document.getElementById('modal');
    box.className = 'modal' + (wide ? ' wide' : '');
    box.innerHTML = html;
    bg.hidden = false;
    return box;
  }
  function closeModal() { document.getElementById('modalBg').hidden = true; }

  function confirm(message, { title = 'Are you sure?', danger = true, okLabel = 'Confirm' } = {}) {
    return new Promise((resolve) => {
      const box = modal(`
        <h3>${esc(title)}</h3>
        <p class="hint" style="margin-bottom:16px">${esc(message)}</p>
        <div class="row" style="justify-content:flex-end">
          <button class="btn" data-x="no">Cancel</button>
          <button class="btn ${danger ? 'danger' : 'primary'}" data-x="yes">${esc(okLabel)}</button>
        </div>`);
      box.querySelector('[data-x=no]').onclick = () => { closeModal(); resolve(false); };
      box.querySelector('[data-x=yes]').onclick = () => { closeModal(); resolve(true); };
    });
  }

  // ── formatting ─────────────────────────────────────────
  function esc(s) {
    return String(s ?? '')
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function bytes(n) {
    if (!n) return '0 B';
    const u = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(n) / Math.log(1024));
    return (n / Math.pow(1024, i)).toFixed(i ? 1 : 0) + ' ' + u[i];
  }
  function ago(ts) {
    if (!ts) return '—';
    const s = Math.floor(Date.now() / 1000 - ts);
    if (s < 60) return s + 's ago';
    if (s < 3600) return Math.floor(s / 60) + 'm ago';
    if (s < 86400) return Math.floor(s / 3600) + 'h ago';
    return Math.floor(s / 86400) + 'd ago';
  }
  function dur(s) {
    if (s == null) return '—';
    if (s < 60) return Math.round(s) + 's';
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m ${Math.round(s % 60)}s`;
    return `${Math.floor(m / 60)}h ${m % 60}m`;
  }

  // ── DOM helpers ────────────────────────────────────────
  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  /** Wire every [data-act] inside a root to a handler map. Keeps tab code free
   *  of per-element addEventListener boilerplate and survives re-renders. */
  function acts(root, map) {
    root.addEventListener('click', (e) => {
      const el = e.target.closest('[data-act]');
      if (!el || !root.contains(el)) return;
      const fn = map[el.dataset.act];
      if (!fn) return;
      e.preventDefault();
      Promise.resolve(fn(el, e)).catch(err);
    });
    root.addEventListener('change', (e) => {
      const el = e.target.closest('[data-change]');
      if (!el || !root.contains(el)) return;
      const fn = map[el.dataset.change];
      if (fn) Promise.resolve(fn(el, e)).catch(err);
    });
  }

  function field(label, inner, note = '') {
    return `<label class="field"><span class="lbl">${esc(label)}${
      note ? ` <span class="note">${esc(note)}</span>` : ''}</span>${inner}</label>`;
  }
  function num(id, value, { min, max, step = 1 } = {}) {
    return `<input type="number" id="${id}" value="${value}"${
      min !== undefined ? ` min="${min}"` : ''}${max !== undefined ? ` max="${max}"` : ''} step="${step}">`;
  }
  function sel(id, options, current) {
    return `<select id="${id}">${options.map((o) => {
      const [v, l] = Array.isArray(o) ? o : [o, o];
      return `<option value="${esc(v)}"${String(v) === String(current) ? ' selected' : ''}>${esc(l)}</option>`;
    }).join('')}</select>`;
  }
  const val = (id) => { const e = document.getElementById(id); return e ? e.value : ''; };
  const numv = (id, d = 0) => { const v = parseFloat(val(id)); return Number.isFinite(v) ? v : d; };
  const chk = (id) => { const e = document.getElementById(id); return e ? e.checked : false; };

  function charOptions(current, { blank = '' } = {}) {
    const chars = (state.boot?.characters || []);
    const opts = blank ? [['', blank]] : [];
    return sel('__x', opts.concat(chars.map(c => [c.id, c.name])), current);
  }

  // ── tabs ───────────────────────────────────────────────
  function tab(name, def) { tabs[name] = def; }

  async function go(name) {
    if (!tabs[name]) return;
    if (state.current && tabs[state.current]?.leave) {
      try { tabs[state.current].leave(); } catch (e) { console.warn(e); }
    }
    state.current = name;
    $$('#rail button').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
    $$('.view').forEach(v => v.classList.toggle('active', v.id === 'view-' + name));
    location.hash = name;
    const root = document.getElementById('view-' + name);
    try {
      if (tabs[name].render) await tabs[name].render(root);
      if (tabs[name].enter) await tabs[name].enter(root);
    } catch (e) { err(e); }
  }

  /** Re-fetch the shared bootstrap payload — call after anything that changes
   *  the set of characters, datasets, adapters or voices. */
  async function refresh() {
    state.boot = await get('/api/bootstrap');
    if (!state.char && state.boot.characters.length) state.char = state.boot.characters[0].id;
    if (!state.dataset && state.boot.datasets.length) state.dataset = state.boot.datasets[0].name;
    if (!state.persona && state.boot.personas.length) state.persona = state.boot.personas[0].name;
    paintEngine(state.boot.engine);
    return state.boot;
  }

  function paintEngine(s) {
    const dot = document.getElementById('engineDot');
    const pill = document.getElementById('enginePill');
    if (!dot) return;
    const st = (s && s.status) || 'stopped';
    dot.className = 'dot ' + (st === 'ready' ? 'ok' : st === 'starting' ? 'busy' : st === 'error' ? 'crit' : '');
    pill.lastChild.textContent = ' Engine · ' + st;
    const stopBtn = document.getElementById('engineStopBtn');
    if (stopBtn) stopBtn.hidden = (st === 'stopped');
    state.engine = s;
  }

  // Quick stop from anywhere in the app — the engine and training fight over
  // the same GPU, so this needs to be reachable without a trip to the Engine
  // tab first.
  const stopEngine = safe(async () => {
    await post('/api/engine/stop');
    ok('Engine stopped');
    schedulePoll(150);
  });

  // ── global status poll (train / generate / engine) ─────
  // Adaptive cadence. A fixed fast poll here plus each tab's own poll opens a
  // new connection every tick; on Windows that exhausts ephemeral ports and
  // every fetch then fails with ERR_NO_BUFFER_SPACE. Poll fast only while
  // something is actually running, and pause entirely in a background tab.
  let pollBusy = false;
  function schedulePoll(ms) {
    clearTimeout(state.poll);
    state.poll = setTimeout(runPoll, ms);
  }
  async function runPoll() {
    if (document.hidden) return schedulePoll(4000);
    if (pollBusy) return schedulePoll(2000);
    pollBusy = true;
    try {
      await pollStatus();
    } finally {
      pollBusy = false;
      const active = state.train?.running || state.train?.generation?.running
        || state.engine?.status === 'starting';
      schedulePoll(active ? 1500 : 5000);
    }
  }

  async function pollStatus() {
    try {
      const [eng, tr] = await Promise.all([
        get('/api/engine/status').catch(() => null),
        get('/api/train/status').catch(() => null),
      ]);
      if (eng) paintEngine(eng);

      const tp = document.getElementById('trainPill');
      if (tr && tr.running) {
        tp.hidden = false;
        tp.className = 'pill accent';
        tp.textContent = `training ${tr.step || 0}/${tr.total_steps || '?'}` +
          (tr.loss != null ? ` · loss ${tr.loss}` : '');
      } else { tp.hidden = true; }

      const gp = document.getElementById('genPill');
      const g = tr && tr.generation;
      if (g && g.running) {
        gp.hidden = false; gp.className = 'pill warn'; gp.textContent = 'generating';
      } else { gp.hidden = true; }

      state.train = tr;
      if (state.current && tabs[state.current]?.onStatus) {
        tabs[state.current].onStatus(tr, eng);
      }
    } catch (e) { /* a dropped poll is not worth a toast */ }
  }

  // ── boot ───────────────────────────────────────────────
  async function boot() {
    const saved = localStorage.getItem('forge-theme');
    if (saved) document.documentElement.dataset.theme = saved;
    document.getElementById('themeBtn').onclick = () => {
      const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
      document.documentElement.dataset.theme = next;
      localStorage.setItem('forge-theme', next);
      if (state.current && tabs[state.current]?.onTheme) tabs[state.current].onTheme();
    };

    document.getElementById('rail').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-tab]');
      if (b) go(b.dataset.tab);
    });
    document.querySelectorAll('[data-go]').forEach(b => {
      b.addEventListener('click', () => go(b.dataset.go));
    });
    document.getElementById('engineStopBtn').addEventListener('click', stopEngine);
    document.getElementById('modalBg').addEventListener('click', (e) => {
      if (e.target.id === 'modalBg') closeModal();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !document.getElementById('modalBg').hidden) closeModal();
    });

    try {
      await refresh();
      document.getElementById('rootPath').textContent =
        (state.boot.project_root || '').split(/[\\/]/).slice(-1)[0];
    } catch (e) {
      err('Could not reach the backend — is app.py running?');
    }

    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) schedulePoll(150);
    });
    runPoll();

    const start = (location.hash || '').replace('#', '');
    go(tabs[start] ? start : 'chat');
  }

  return {
    boot, tab, go, refresh, state, api, get, post, put, del,
    toast, ok, err, safe, modal, closeModal, confirm,
    esc, bytes, ago, dur, $, $$, acts, field, num, sel, val, numv, chk,
    charOptions, paintEngine,
  };
})();
