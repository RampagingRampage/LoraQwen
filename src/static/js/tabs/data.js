/* data.js — the Data tab.
   Generation (with the supervision slider), the review queue, the cleanup
   toolkit, row editing, and DPO ranking. */

Forge.tab('data', (() => {
  let mode = 'browse';         // browse | generate | review | clean | rank
  let rows = [];
  let total = 0;
  let offset = 0;
  let selected = new Set();
  let report = null;
  let queue = [];
  let rank = null;
  let genTimer = null;

  const LIMIT = 60;
  const name = () => Forge.state.dataset;

  function render(root) {
    const dsets = Forge.state.boot?.datasets || [];
    root.innerHTML = `
      <h1 class="page">Data</h1>
      <p class="page-sub">Generate training samples, supervise as much of that as you want,
        then clean what came out. Small and good beats big and noisy — this tab is
        where that gets decided.</p>

      <div class="row" style="margin-bottom:16px">
        <select id="dsSel" style="max-width:220px">
          ${dsets.map(d => `<option value="${d.name}"${d.name === name() ? ' selected' : ''}>${
            Forge.esc(d.name)} · ${d.rows}</option>`).join('') || '<option value="">(none yet)</option>'}
        </select>
        <button class="btn sm" data-act="newds">New</button>
        <button class="btn sm" data-act="import">Import file</button>
        <button class="btn sm" data-act="export">Export</button>
        <div class="spacer"></div>
        <div class="row tight" id="modeBar">
          ${['browse', 'generate', 'review', 'clean', 'rank'].map(m =>
            `<button class="btn sm${m === mode ? ' primary' : ''}" data-act="mode" data-m="${m}">${
              m === 'rank' ? 'DPO ranking' : m[0].toUpperCase() + m.slice(1)}</button>`).join('')}
        </div>
      </div>

      <div id="dataBody"></div>`;

    Forge.acts(root, handlers);
    document.getElementById('dsSel').onchange = Forge.safe(async (e) => {
      Forge.state.dataset = e.target.value;
      offset = 0; selected.clear(); report = null;
      await paint();
    });
  }

  const handlers = {
    mode: Forge.safe(async (el) => { mode = el.dataset.m; offset = 0; await paint(); render(document.getElementById('view-data')); await paint(); }),
    newds: () => newDataset(),
    import: () => importFile(),
    export: () => { if (name()) location.href = `/api/dataset/${name()}/export`; },

    // browse
    page: Forge.safe(async (el) => { offset = Math.max(0, offset + (+el.dataset.d) * LIMIT); await paint(); }),
    search: Forge.safe(async () => { offset = 0; await paint(); }),
    toggleRow: (el) => {
      const i = +el.dataset.i;
      selected.has(i) ? selected.delete(i) : selected.add(i);
      el.closest('.drow').classList.toggle('sel', selected.has(i));
      paintBulk();
    },
    selectAll: Forge.safe(async () => {
      rows.forEach(r => selected.add(r.idx));
      await paint();
    }),
    selectNone: Forge.safe(async () => { selected.clear(); await paint(); }),
    edit: (el) => editRow(+el.dataset.i),
    regen: (el) => regenRow(+el.dataset.i),
    delSelected: Forge.safe(async () => {
      if (!selected.size) return;
      if (!await Forge.confirm(`Deletes ${selected.size} rows. A snapshot is taken first, so this is undoable from the Clean tab.`,
        { title: 'Delete selected rows?', okLabel: 'Delete' })) return;
      const r = await Forge.del(`/api/dataset/${name()}/rows`, { indices: [...selected] });
      Forge.ok(`Removed ${r.removed} rows`);
      selected.clear();
      await Forge.refresh(); await paint();
    }),

    // generate
    startGen: () => startGeneration(),
    stopGen: Forge.safe(async () => { await Forge.post('/api/generate/stop'); Forge.ok('Stopping…'); }),
    probeLM: Forge.safe(async () => {
      const r = await Forge.post('/api/generate/probe', { url: Forge.val('lmUrl') });
      if (!r.ok) return Forge.err(`Nothing there: ${r.error}`);
      const s = document.getElementById('lmModel');
      s.innerHTML = r.models.map(m => `<option>${Forge.esc(m)}</option>`).join('');
      Forge.ok(`${r.models.length} model(s) available`);
    }),
    srcChange: () => paintGenSource(),
    uploadBook: () => uploadBookDialog(),
    deleteBook: () => deleteBook(),
    collectClaude: Forge.safe(async () => {
      const r = await Forge.post(`/api/generate/claude/${name()}/collect`,
        { supervise_pct: Forge.numv('supPct', 0) });
      Forge.ok(`Imported ${r.added} · ${r.held} held for review · ${r.rejected} rejected`);
      await Forge.refresh(); await paint();
    }),

    // review
    decide: (el) => decide(el.dataset.id, el.dataset.d),
    regenReview: (el) => regenReview(el.dataset.id),

    // clean
    analyze: () => analyze(),
    applyClean: () => applyClean(),
    restore: (el) => restore(el.dataset.f),

    // rank
    draw: () => drawCandidates(),
    pick: (el) => pickCandidate(+el.dataset.i),
    pickCustom: () => pickCustom(),
    runDpo: Forge.safe(async () => {
      if (!await Forge.confirm('Continues the existing adapter with your preference pairs. Training takes over the GPU.',
        { title: 'Run DPO pass?', danger: false, okLabel: 'Run' })) return;
      await Forge.post('/api/train/dpo', { name: name() });
      Forge.ok('DPO training started — watch it in the Train tab');
      Forge.go('train');
    }),
  };

  // ── paint dispatch ─────────────────────────────────────
  async function paint() {
    const body = document.getElementById('dataBody');
    if (!body) return;
    if (!name()) {
      body.innerHTML = `<div class="empty"><div class="big">▤</div>
        <p><strong>No dataset selected.</strong></p>
        <p>Create one, import a file, or generate from a persona.</p></div>`;
      return;
    }
    if (mode === 'browse') return paintBrowse(body);
    if (mode === 'generate') return paintGenerate(body);
    if (mode === 'review') return paintReview(body);
    if (mode === 'clean') return paintClean(body);
    if (mode === 'rank') return paintRank(body);
  }

  // ── BROWSE ─────────────────────────────────────────────
  async function paintBrowse(body) {
    const q = Forge.val('dsSearch');
    const d = await Forge.get(`/api/dataset/${name()}?offset=${offset}&limit=${LIMIT}${
      q ? '&q=' + encodeURIComponent(q) : ''}`);
    rows = d.rows; total = d.total;

    body.innerHTML = `
      <div class="row" style="margin-bottom:12px">
        <input type="search" id="dsSearch" placeholder="Filter rows…" value="${Forge.esc(q)}" style="max-width:280px">
        <button class="btn sm" data-act="search">Filter</button>
        <span class="pill">${total} rows</span>
        ${d.queue.pending ? `<span class="pill warn">${d.queue.pending} awaiting review</span>` : ''}
        ${d.bad_lines.length ? `<span class="pill crit">${d.bad_lines.length} unparseable lines</span>` : ''}
        <div class="spacer"></div>
        <div id="bulkBar"></div>
      </div>
      <div id="rowsBox">${rows.map(rowHtml).join('') || '<div class="empty"><p>No rows here.</p></div>'}</div>
      <div class="row" style="margin-top:14px;justify-content:center">
        <button class="btn sm" data-act="page" data-d="-1" ${offset ? '' : 'disabled'}>← Previous</button>
        <span class="pill">${offset + 1}–${Math.min(offset + LIMIT, total)} of ${total}</span>
        <button class="btn sm" data-act="page" data-d="1" ${offset + LIMIT < total ? '' : 'disabled'}>Next →</button>
      </div>`;
    document.getElementById('dsSearch').addEventListener('keydown', e => {
      if (e.key === 'Enter') handlers.search();
    });
    paintBulk();
  }

  function rowHtml(r) {
    const tok = Math.round((r.instruction.length + r.output.length) / 3.6);
    return `
      <div class="drow${selected.has(r.idx) ? ' sel' : ''}">
        <input type="checkbox" data-act="toggleRow" data-i="${r.idx}"${selected.has(r.idx) ? ' checked' : ''}>
        <div>
          <div class="q">${Forge.esc(r.instruction)}</div>
          <div class="a">${Forge.esc(r.output)}</div>
          ${r.reasoning ? `<div class="a" style="opacity:.6;font-style:italic;margin-top:4px">think: ${
            Forge.esc(r.reasoning.slice(0, 180))}</div>` : ''}
          <div class="flags">
            <span class="pill">#${r.idx}</span>
            <span class="pill">~${tok} tok</span>
            ${r.reasoning ? '<span class="pill accent">thinking</span>' : ''}
          </div>
        </div>
        <div class="acts">
          <button class="btn sm" data-act="edit" data-i="${r.idx}">Edit</button>
          <button class="btn sm" data-act="regen" data-i="${r.idx}">Regenerate</button>
        </div>
      </div>`;
  }

  function paintBulk() {
    const el = document.getElementById('bulkBar');
    if (!el) return;
    el.innerHTML = selected.size
      ? `<div class="row tight">
           <span class="pill accent">${selected.size} selected</span>
           <button class="btn sm" data-act="selectNone">Clear</button>
           <button class="btn sm danger" data-act="delSelected">Delete selected</button>
         </div>`
      : `<button class="btn sm" data-act="selectAll">Select page</button>`;
  }

  function editRow(idx) {
    const r = rows.find(x => x.idx === idx);
    if (!r) return;
    const box = Forge.modal(`
      <h3>Edit row #${idx}</h3>
      <p class="hint" style="margin-bottom:12px">A snapshot is taken before the file is rewritten.</p>
      ${Forge.field('Prompt', `<textarea id="eI" rows="3">${Forge.esc(r.instruction)}</textarea>`)}
      ${Forge.field('Response', `<textarea id="eO" rows="6">${Forge.esc(r.output)}</textarea>`)}
      ${Forge.field('Reasoning', `<textarea id="eR" rows="3" placeholder="Optional — only used by the thinking format.">${
        Forge.esc(r.reasoning || '')}</textarea>`)}
      <p class="hint" id="tokCount"></p>
      <div class="row" style="justify-content:flex-end;margin-top:12px">
        <button class="btn" data-x="c">Cancel</button>
        <button class="btn primary" data-x="s">Save</button></div>`, { wide: true });

    const count = () => {
      const n = Math.round((box.querySelector('#eI').value.length +
        box.querySelector('#eO').value.length + box.querySelector('#eR').value.length) / 3.6);
      box.querySelector('#tokCount').innerHTML = `~${n} tokens` +
        (n > 1024 ? ` <span class="pill crit">over 1024 — will be truncated at the default seq length</span>` : '');
    };
    box.querySelectorAll('textarea').forEach(t => t.addEventListener('input', count));
    count();

    box.querySelector('[data-x=c]').onclick = Forge.closeModal;
    box.querySelector('[data-x=s]').onclick = Forge.safe(async () => {
      await Forge.put(`/api/dataset/${name()}/row/${idx}`, {
        instruction: box.querySelector('#eI').value,
        output: box.querySelector('#eO').value,
        reasoning: box.querySelector('#eR').value,
      });
      Forge.closeModal(); Forge.ok('Row saved'); await paint();
    });
  }

  async function regenRow(idx) {
    const r = rows.find(x => x.idx === idx);
    if (!r) return;
    Forge.toast('Regenerating…');
    const res = await Forge.post(`/api/dataset/${name()}/regenerate`, { index: idx, apply: false });
    const box = Forge.modal(`
      <h3>Regenerated answer</h3>
      ${Forge.field('Prompt (unchanged)', `<textarea rows="2" readonly>${Forge.esc(r.instruction)}</textarea>`)}
      ${Forge.field('Current', `<textarea rows="4" readonly>${Forge.esc(r.output)}</textarea>`)}
      ${Forge.field('New', `<textarea id="newOut" rows="4">${Forge.esc(res.output)}</textarea>`)}
      <div class="row" style="justify-content:flex-end;margin-top:12px">
        <button class="btn" data-x="c">Keep current</button>
        <button class="btn" data-x="a">Try again</button>
        <button class="btn primary" data-x="s">Use new</button></div>`, { wide: true });
    box.querySelector('[data-x=c]').onclick = Forge.closeModal;
    box.querySelector('[data-x=a]').onclick = () => { Forge.closeModal(); regenRow(idx); };
    box.querySelector('[data-x=s]').onclick = Forge.safe(async () => {
      await Forge.put(`/api/dataset/${name()}/row/${idx}`, { output: box.querySelector('#newOut').value });
      Forge.closeModal(); Forge.ok('Row replaced'); await paint();
    });
  }

  // ── GENERATE ───────────────────────────────────────────
  function paintGenerate(body) {
    const g = Forge.state.boot?.defaults?.generation || {};
    const personas = Forge.state.boot?.personas || [];
    body.innerHTML = `
      <div class="grid c2">
        <div class="card">
          <div class="card-head"><h3>Source</h3></div>
          ${Forge.field('Who writes the samples', Forge.sel('genSrc', [
            ['engine', 'Local engine — the running base model'],
            ['lmstudio', 'LM Studio (or any OpenAI-compatible endpoint)'],
            ['claude', 'Claude Code — via a file handshake'],
            ['book', 'Book extraction — real dialogue from text you provide'],
          ], 'engine').replace('id="__x"', 'id="genSrc" data-change="srcChange"'))}
          <div id="srcExtra"></div>

          <div id="personaFields">
            ${Forge.field('Persona profile', Forge.sel('genPersona',
              personas.map(p => p.name), name()).replace('id="__x"', 'id="genPersona"'),
              'supplies the topics and voice')}
            ${Forge.field('Persona override', `<textarea id="genOverride" rows="3"
              placeholder="Optional — paste a persona summary to use instead of the saved profile."></textarea>`)}
          </div>
        </div>

        <div class="card" id="settingsCard">
          <div class="card-head"><h3>Settings</h3></div>
          <div class="grid c2" style="gap:0 12px" id="genSettingsGrid">
            ${Forge.field('Target total', Forge.num('genTotal', g.total ?? 300, { min: 4 }))}
            ${Forge.field('Batch size', Forge.num('genBatch', g.batch_size ?? 8, { min: 1, max: 32 }))}
            ${Forge.field('Temperature', Forge.num('genTemp', g.temperature ?? 0.9, { min: 0, max: 2, step: 0.05 }))}
            ${Forge.field('Timeout (s)', Forge.num('genTimeout', g.timeout_s ?? 240, { min: 30 }))}
            ${Forge.field('Sample format', Forge.sel('genFmt', [
              ['instruct', 'Instruct — prompt → reply'],
              ['thinking', 'Thinking — reasoning + reply'],
              ['chat', 'Chat — multi-turn messages'],
            ], 'instruct').replace('id="__x"', 'id="genFmt"'))}
            ${Forge.field('Dedupe threshold', Forge.num('genDedupe', g.dedupe_threshold ?? 0.62,
              { min: 0.3, max: 0.99, step: 0.01 }), 'lower = stricter')}
          </div>
          <label class="check"><input type="checkbox" id="genFresh"> Start fresh (snapshot then clear the dataset)</label>
        </div>
      </div>

      <div class="card">
        <div class="card-head"><h3>Supervision</h3>
          <div class="spacer"></div>
          <span class="pill accent" id="supLabel">0% — fully automatic</span></div>
        <p class="hint" style="margin-bottom:12px">
          What fraction of generated samples is held for you to look at before it
          enters the dataset. The rest streams straight in while you review.</p>
        <input type="range" id="supPct" min="0" max="100" step="5" value="${g.supervise_pct ?? 0}">
        <div class="row" style="justify-content:space-between;font-size:11px;color:var(--faint);margin-top:2px">
          <span>0 — trust the generator</span><span>50</span><span>100 — review every one</span>
        </div>
      </div>

      <div class="row">
        <button class="btn primary" data-act="startGen">Start generating</button>
        <button class="btn" data-act="stopGen">Stop</button>
      </div>

      <div class="card" id="genProgress" style="margin-top:14px"></div>`;

    const s = document.getElementById('supPct');
    const lbl = document.getElementById('supLabel');
    const setLbl = () => {
      const v = +s.value;
      lbl.textContent = v === 0 ? '0% — fully automatic'
        : v === 100 ? '100% — review every sample'
        : `${v}% — about 1 in ${Math.round(100 / v)} held for review`;
    };
    s.oninput = setLbl; setLbl();
    paintGenSource();
    pollGen();
  }

  let bookChunkTimer = null;

  function paintGenSource() {
    const src = Forge.val('genSrc');
    const box = document.getElementById('srcExtra');
    const personaFields = document.getElementById('personaFields');
    const settingsGrid = document.getElementById('genSettingsGrid');
    if (!box) return;
    personaFields.hidden = src === 'book';
    if (src === 'lmstudio') {
      box.innerHTML = `
        ${Forge.field('Endpoint', `<input type="text" id="lmUrl" class="mono"
          value="${Forge.esc(Forge.state.boot?.defaults?.lmstudio_url || 'http://localhost:1234')}">`)}
        <div class="row" style="margin:-6px 0 12px">
          <button class="btn sm" data-act="probeLM">Check connection</button></div>
        ${Forge.field('Model', `<select id="lmModel"><option value="">(check connection first)</option></select>`)}`;
    } else if (src === 'claude') {
      box.innerHTML = `
        <p class="hint" style="margin:0 0 10px">
          Starting a run writes a self-contained request spec to
          <code>datasets/_requests/</code>. Point Claude Code at it, let it write
          the answers to <code>datasets/_responses/</code>, then collect them here.
          No API key, no network call from this app.</p>
        <button class="btn sm" data-act="collectClaude">Collect Claude output</button>`;
    } else if (src === 'book') {
      box.innerHTML = `
        <p class="hint" style="margin:0 0 10px">
          Pulls real dialogue and narrated inner thought straight out of a book's
          own text, instead of inventing new scenarios. Works best on a
          first-person narrator — only they have inner thought on the page to
          extract. Upload text you already have the rights to; this never
          fetches anything on its own.</p>
        ${Forge.field('Book', `<select id="bookFile"></select>`)}
        <div class="row" style="margin:-6px 0 10px">
          <button class="btn sm" data-act="uploadBook">Upload a book…</button>
          <button class="btn sm" data-act="deleteBook">Delete</button>
        </div>
        ${Forge.field('Protagonist', `<input type="text" id="bookProtagonist"
          placeholder="the name the extractor should pull lines for">`)}
        ${Forge.field('Extracted by', Forge.sel('bookSrc', [
          ['engine', 'Local engine'], ['lmstudio', 'LM Studio'],
        ], 'engine').replace('id="__x"', 'id="bookSrc"'))}
        ${Forge.field('Chunk size (characters)', Forge.num('chunkChars', 3000, { min: 800, max: 8000, step: 200 }),
          'bigger = more context per call, fewer calls')}
        ${Forge.field('Limit to first N chunks', Forge.num('maxChunks', 0, { min: 0 }), '0 = the whole book')}
        <div id="bookPreview" class="hint" style="margin:8px 0 0"></div>`;
      loadBookList();
    } else {
      box.innerHTML = `<p class="hint" style="margin:0 0 12px">
        Uses the model already loaded in the engine. The base model writes the
        samples; the persona profile tells it whose voice to write in.</p>`;
    }
    // Row target / batch size / dedupe-as-usual don't apply to book mode --
    // it processes every chunk once, driven by chunk count, not a row goal.
    if (settingsGrid) settingsGrid.style.opacity = src === 'book' ? '.4' : '';
    Forge.$$('#genSettingsGrid input, #genSettingsGrid select').forEach(el => {
      el.disabled = src === 'book';
    });
  }

  async function loadBookList() {
    const books = await Forge.get(`/api/persona/${Forge.val('genPersona') || name()}/books`).catch(() => []);
    const sel = document.getElementById('bookFile');
    if (!sel) return;
    sel.innerHTML = books.length
      ? books.map(b => `<option value="${Forge.esc(b.file)}">${Forge.esc(b.file)} · ${
          Forge.bytes(b.bytes)} · ${b.chapters || '?'} chapters</option>`).join('')
      : '<option value="">(none uploaded yet)</option>';
    schedulePreview();
  }

  function schedulePreview() {
    clearTimeout(bookChunkTimer);
    bookChunkTimer = setTimeout(previewBook, 400);
  }

  async function previewBook() {
    const persona = Forge.val('genPersona') || name();
    const file = Forge.val('bookFile');
    const box = document.getElementById('bookPreview');
    if (!file || !box) { if (box) box.textContent = ''; return; }
    box.textContent = 'checking…';
    try {
      const p = await Forge.post(`/api/persona/${persona}/books/${encodeURIComponent(file)}/preview`,
        { chunk_chars: Forge.numv('chunkChars', 3000) });
      box.innerHTML = `${p.chapters} chapter(s) detected · <strong>${p.chunks} chunks</strong> to process · ` +
        `${Forge.bytes(p.chars)} of text · roughly ${p.chunks} model calls at the current chunk size`;
    } catch (e) { box.textContent = ''; }
  }

  function uploadBookDialog() {
    const persona = Forge.val('genPersona') || name();
    const box = Forge.modal(`
      <h3>Upload a book</h3>
      <p class="hint" style="margin-bottom:12px">Plain text, from something you already have the
        rights to — an export from your own ebook library, or pasted text. This never fetches
        anything from the internet on its own.</p>
      ${Forge.field('File (.txt)', `<input type="file" id="bkFile" accept=".txt,.md">`)}
      <p class="hint" style="margin:8px 0 4px">— or —</p>
      ${Forge.field('Paste text', `<textarea id="bkText" rows="6" placeholder="Paste the book's text here…"></textarea>`)}
      ${Forge.field('Save as', `<input type="text" id="bkName" placeholder="book-title.txt">`)}
      <div class="row" style="justify-content:flex-end;margin-top:12px">
        <button class="btn" data-x="c">Cancel</button>
        <button class="btn primary" data-x="s">Save</button></div>`);
    box.querySelector('[data-x=c]').onclick = Forge.closeModal;
    box.querySelector('[data-x=s]').onclick = Forge.safe(async () => {
      const f = box.querySelector('#bkFile').files[0];
      const pasted = box.querySelector('#bkText').value;
      const filename = box.querySelector('#bkName').value.trim() ||
        (f ? f.name : 'book.txt');
      if (f) {
        const fd = new FormData();
        fd.append('file', f);
        fd.append('filename', filename);
        await Forge.api(`/api/persona/${persona}/books`, { method: 'POST', body: fd });
      } else if (pasted.trim()) {
        const fd = new FormData();
        fd.append('text', pasted);
        fd.append('filename', filename);
        await Forge.api(`/api/persona/${persona}/books`, { method: 'POST', body: fd });
      } else {
        return Forge.err('Pick a file or paste some text.');
      }
      Forge.closeModal();
      Forge.ok('Book saved');
      await loadBookList();
    });
  }

  async function deleteBook() {
    const persona = Forge.val('genPersona') || name();
    const file = Forge.val('bookFile');
    if (!file) return;
    if (!await Forge.confirm(`Deletes ${file} from this persona's uploaded books.`,
      { title: 'Delete book?', okLabel: 'Delete' })) return;
    await Forge.del(`/api/persona/${persona}/books/${encodeURIComponent(file)}`);
    Forge.ok('Deleted');
    await loadBookList();
  }

  async function startGeneration() {
    const cfg = {
      name: name(),
      source: Forge.val('genSrc'),
      lmstudio_url: Forge.val('lmUrl') || undefined,
      model: Forge.val('lmModel') || undefined,
      persona: Forge.val('genPersona'),
      persona_override: Forge.val('genOverride') || undefined,
      total: Forge.numv('genTotal', 300),
      batch_size: Forge.numv('genBatch', 8),
      temperature: Forge.numv('genTemp', 0.9),
      timeout_s: Forge.numv('genTimeout', 240),
      format: Forge.val('genFmt'),
      dedupe_threshold: Forge.numv('genDedupe', 0.62),
      supervise_pct: Forge.numv('supPct', 0),
      fresh: Forge.chk('genFresh'),
    };
    await Forge.post('/api/generate/start', { ...cfg, persona_name: cfg.persona });
    Forge.ok('Generation started');
    pollGen();
  }

  function pollGen() {
    clearInterval(genTimer);
    genTimer = setInterval(async () => {
      const box = document.getElementById('genProgress');
      if (!box) return clearInterval(genTimer);
      let g;
      try { g = await Forge.get('/api/generate/status'); } catch { return; }
      const done = g.added + g.held;
      const pct = g.target ? Math.min(100, Math.round(done / g.target * 100)) : 0;
      const heldPct = g.target ? Math.round(g.held / g.target * 100) : 0;
      box.innerHTML = `
        <div class="card-head">
          <h3>${g.status === 'idle' ? 'Not running' : g.status.replace('_', ' ')}</h3>
          <div class="spacer"></div>
          ${g.running ? '<span class="dot busy"></span>' : ''}
        </div>
        <div class="grid c4" style="margin-bottom:12px">
          <div class="stat"><div class="k">In dataset</div><div class="v">${g.added}</div></div>
          <div class="stat"><div class="k">Held for review</div><div class="v" style="color:var(--warn)">${g.held}</div></div>
          <div class="stat"><div class="k">Rejected</div><div class="v sm">${g.rejected}</div><div class="n">duplicates</div></div>
          <div class="stat"><div class="k">Batch</div><div class="v sm">${g.batch}/${g.batches || '?'}</div></div>
        </div>
        <div class="bar split"><i style="width:${pct - heldPct}%"></i></div>
        <p class="hint">${done} of ${g.target} · ${pct}%</p>
        ${g.error ? `<p class="hint" style="color:var(--crit)">${Forge.esc(g.error)}</p>` : ''}
        <div class="log" style="margin-top:10px">${(g.log || []).slice(-40).map(Forge.esc).join('\n')}</div>`;
      if (!g.running && g.status !== 'generating') clearInterval(genTimer);
    }, 2000);
  }

  // ── REVIEW ─────────────────────────────────────────────
  async function paintReview(body) {
    const d = await Forge.get(`/api/review/${name()}`);
    queue = d.items;
    const s = d.stats;
    body.innerHTML = `
      <div class="grid c4" style="margin-bottom:16px">
        <div class="stat"><div class="k">Pending</div><div class="v">${s.pending}</div></div>
        <div class="stat"><div class="k">Accepted</div><div class="v" style="color:var(--ok)">${s.accepted}</div></div>
        <div class="stat"><div class="k">Rejected</div><div class="v" style="color:var(--crit)">${s.rejected}</div></div>
        <div class="stat"><div class="k">Held total</div><div class="v sm">${s.held_total}</div></div>
      </div>
      ${!queue.length ? `<div class="empty"><div class="big">✓</div>
        <p><strong>Nothing waiting.</strong></p>
        <p>Samples land here when you generate with the supervision slider above zero.</p></div>`
      : queue.map(reviewHtml).join('')}`;
  }

  function reviewHtml(it) {
    const r = it.record;
    const instr = r.instruction || (r.messages?.[0]?.content) || '';
    const out = r.output || (r.messages?.[1]?.content) || '';
    return `
      <div class="card" data-review="${it.id}">
        <div class="card-head">
          <span class="pill">${Forge.esc(it.meta.source || '?')}</span>
          ${it.meta.temperature ? `<span class="pill">temp ${it.meta.temperature}</span>` : ''}
          ${it.meta.format ? `<span class="pill">${Forge.esc(it.meta.format)}</span>` : ''}
          <div class="spacer"></div>
          <span class="pill">${Forge.ago(it.ts)}</span>
        </div>
        ${Forge.field('Prompt', `<textarea rows="2" id="ri-${it.id}">${Forge.esc(instr)}</textarea>`)}
        ${Forge.field('Response', `<textarea rows="4" id="ro-${it.id}">${Forge.esc(out)}</textarea>`)}
        <div class="row">
          <button class="btn primary sm" data-act="decide" data-id="${it.id}" data-d="accept">Accept</button>
          <button class="btn sm" data-act="decide" data-id="${it.id}" data-d="edit">Save edits &amp; accept</button>
          <button class="btn sm" data-act="regenReview" data-id="${it.id}">Regenerate</button>
          <div class="spacer"></div>
          <button class="btn danger sm" data-act="decide" data-id="${it.id}" data-d="reject">Reject</button>
        </div>
      </div>`;
  }

  async function decide(id, action) {
    const payload = { id, action };
    if (action === 'edit') {
      payload.record = {
        instruction: document.getElementById('ri-' + id).value,
        input: '',
        output: document.getElementById('ro-' + id).value,
      };
    }
    await Forge.post(`/api/review/${name()}/decide`, payload);
    document.querySelector(`[data-review="${id}"]`)?.remove();
    await Forge.refresh();
    if (!document.querySelector('[data-review]')) await paint();
  }

  async function regenReview(id) {
    const r = await Forge.post(`/api/review/${name()}/decide`, { id, action: 'regenerate' });
    document.getElementById('ro-' + id).value = r.output;
    Forge.ok('Regenerated — accept it or try again');
  }

  // ── CLEAN ──────────────────────────────────────────────
  async function paintClean(body) {
    const snaps = await Forge.get(`/api/dataset/${name()}/snapshots`).catch(() => []);
    body.innerHTML = `
      <div class="card">
        <div class="card-head"><h3>Analyze</h3><div class="spacer"></div>
          <button class="btn sm primary" data-act="analyze">Run analysis</button></div>
        <div class="grid c3" style="gap:0 12px">
          ${Forge.field('Duplicate threshold', Forge.num('clDupe', 0.62, { min: 0.3, max: 0.99, step: 0.01 }),
            'lower = stricter')}
          ${Forge.field('Max sequence length', Forge.num('clSeq', 1024, { min: 128, step: 128 }),
            'must match training')}
          ${Forge.field('Compare style against', Forge.sel('clPersona',
            [['', '(skip drift check)']].concat((Forge.state.boot?.personas || []).map(p => p.name)), name())
            .replace('id="__x"', 'id="clPersona"'))}
        </div>
        <div id="reportBox"></div>
      </div>

      <div class="card">
        <div class="card-head"><h3>Clean</h3></div>
        <p class="hint" style="margin-bottom:10px">Every operation snapshots the file first.</p>
        <div class="grid c2" style="gap:0 20px">
          <div>
            <label class="check"><input type="checkbox" id="op_drop_duplicates" checked> Drop near-duplicates</label>
            <label class="check"><input type="checkbox" id="op_drop_empty" checked> Drop empty / malformed rows</label>
            <label class="check"><input type="checkbox" id="op_drop_too_short"> Drop very short answers</label>
            <label class="check"><input type="checkbox" id="op_drop_too_long"> Drop rows over the sequence length</label>
          </div>
          <div>
            <label class="check"><input type="checkbox" id="op_strip_urls"> Strip URLs</label>
            <label class="check"><input type="checkbox" id="op_strip_phones"> Strip phone numbers</label>
            <label class="check"><input type="checkbox" id="op_strip_emails"> Strip email addresses</label>
            <label class="check"><input type="checkbox" id="op_strip_handles"> Strip @handles</label>
            <label class="check"><input type="checkbox" id="op_trim_whitespace" checked> Trim whitespace</label>
          </div>
        </div>
        <div class="row" style="margin-top:12px">
          <button class="btn primary" data-act="applyClean">Apply cleanup</button>
        </div>
      </div>

      <div class="card">
        <div class="card-head"><h3>Snapshots</h3></div>
        <p class="hint" style="margin-bottom:10px">Taken automatically before every destructive edit. The last 20 are kept.</p>
        ${snaps.length ? `<div class="tw"><table><thead><tr><th>File</th><th class="num">Size</th><th class="num">When</th><th></th></tr></thead><tbody>
          ${snaps.map(s => `<tr><td class="mono">${Forge.esc(s.file)}</td>
            <td class="num">${Forge.bytes(s.bytes)}</td><td class="num">${Forge.ago(s.modified)}</td>
            <td><button class="btn sm" data-act="restore" data-f="${Forge.esc(s.file)}">Restore</button></td></tr>`).join('')}
        </tbody></table></div>` : '<p class="hint">None yet.</p>'}`;
  }

  async function analyze() {
    Forge.toast('Analyzing…');
    report = await Forge.post(`/api/dataset/${name()}/analyze`, {
      dupe_threshold: Forge.numv('clDupe', 0.62),
      max_seq_length: Forge.numv('clSeq', 1024),
      persona: Forge.val('clPersona') || undefined,
    });
    const s = report.summary;
    const chip = (label, n, kind) =>
      `<div class="stat"><div class="k">${label}</div><div class="v ${n ? '' : 'sm'}"${
        n && kind ? ` style="color:var(--${kind})"` : ''}>${n}</div></div>`;
    document.getElementById('reportBox').innerHTML = `
      <h3 class="sub">Findings across ${report.total} rows</h3>
      <div class="grid c4">
        ${chip('Near-duplicates', s.duplicates, 'warn')}
        ${chip('Over length', s.too_long, 'warn')}
        ${chip('Too short', s.too_short, 'warn')}
        ${chip('Empty', s.empty, 'crit')}
        ${chip('URLs / phones / emails', s.pii, 'warn')}
        ${chip('Style drift', s.drift, 'warn')}
        ${chip('Unparseable', s.unparseable, 'crit')}
        ${chip('Clean rows', report.clean_rows, 'ok')}
      </div>
      ${s.no_reasoning === report.total && report.total ? `<p class="hint" style="margin-top:10px">
        No rows carry a reasoning span, so a model trained on this won't learn to
        think. That's fine for a voice adapter — it only matters if you wanted a
        reasoning one.</p>` : ''}`;
  }

  async function applyClean() {
    const ops = ['drop_duplicates', 'drop_empty', 'drop_too_short', 'drop_too_long',
      'strip_urls', 'strip_phones', 'strip_emails', 'strip_handles', 'trim_whitespace']
      .filter(o => Forge.chk('op_' + o));
    if (!ops.length) return Forge.err('Pick at least one operation.');
    if (!await Forge.confirm(`Applies ${ops.length} operation(s) and rewrites the file. A snapshot is taken first.`,
      { title: 'Clean dataset?', danger: false, okLabel: 'Clean' })) return;
    const r = await Forge.post(`/api/dataset/${name()}/clean`, {
      ops, dupe_threshold: Forge.numv('clDupe', 0.62), max_seq_length: Forge.numv('clSeq', 1024),
    });
    Forge.ok(`${r.before} → ${r.after} rows (${r.before - r.after} removed, ${r.scrubbed} scrubbed)`);
    await Forge.refresh(); await paint();
  }

  async function restore(f) {
    if (!await Forge.confirm(`Replaces the current dataset with ${f}. The current file is snapshotted first.`,
      { title: 'Restore snapshot?', okLabel: 'Restore' })) return;
    await Forge.post(`/api/dataset/${name()}/restore`, { snapshot: f });
    Forge.ok('Restored');
    await Forge.refresh(); await paint();
  }

  // ── DPO RANKING ────────────────────────────────────────
  async function paintRank(body) {
    const s = await Forge.get(`/api/dpo/${name()}/stats`);
    body.innerHTML = `
      <div class="grid c4" style="margin-bottom:16px">
        <div class="stat"><div class="k">Rounds ranked</div><div class="v">${s.rounds}</div></div>
        <div class="stat"><div class="k">Preference pairs</div><div class="v">${s.pairs}</div></div>
        <div class="stat"><div class="k">Ready for DPO</div>
          <div class="v sm" style="color:var(--${s.ready ? 'ok' : 'dim'})">${s.ready ? 'yes' : 'not yet'}</div>
          <div class="n">needs 20+ pairs</div></div>
        <div class="stat"><div class="k">Character</div><div class="v sm">${
          Forge.esc((Forge.state.boot.characters.find(c => c.id === Forge.state.char) || {}).name || '—')}</div></div>
      </div>

      <div class="card">
        <div class="card-head"><h3>Rank responses</h3><div class="spacer"></div>
          <select id="rankChar" style="max-width:180px">${(Forge.state.boot.characters || []).map(c =>
            `<option value="${c.id}"${c.id === Forge.state.char ? ' selected' : ''}>${Forge.esc(c.name)}</option>`).join('')}</select>
          <button class="btn primary sm" data-act="draw">Draw a prompt</button></div>
        <p class="hint">Four answers to the same training prompt at spread temperatures.
          Pick the one that sounds most right, or write your own. Every pick becomes
          three chosen/rejected pairs.</p>
        <div id="rankBox"></div>
      </div>

      <div class="row">
        <button class="btn" data-act="runDpo" ${s.ready ? '' : 'disabled'}>Run DPO pass</button>
        ${s.ready ? '' : '<span class="hint">Rank a few more rounds first.</span>'}
      </div>`;
  }

  async function drawCandidates() {
    Forge.state.char = Forge.val('rankChar') || Forge.state.char;
    document.getElementById('rankBox').innerHTML = '<p class="hint">Generating four candidates…</p>';
    rank = await Forge.post(`/api/dpo/${name()}/candidates`, { char_id: Forge.state.char });
    document.getElementById('rankBox').innerHTML = `
      <div class="card tight" style="background:var(--panel-2);margin:12px 0">
        <strong style="font-size:13px">${Forge.esc(rank.prompt)}</strong></div>
      <div class="grid c2">
        ${rank.candidates.map((c, i) => `
          <div class="card tight">
            <div class="card-head"><span class="pill">temp ${c.temperature}</span><div class="spacer"></div>
              <button class="btn sm primary" data-act="pick" data-i="${i}">Pick this</button></div>
            <div style="font-size:13px;white-space:pre-wrap">${Forge.esc(c.text || c.error || '(empty)')}</div>
          </div>`).join('')}
      </div>
      ${Forge.field('None of them? Write it yourself', `<textarea id="customPick" rows="3"></textarea>`)}
      <button class="btn" data-act="pickCustom">Use my version</button>`;
  }

  async function pickCandidate(i) {
    await submitPick(rank.candidates[i].text, false);
  }
  async function pickCustom() {
    const t = Forge.val('customPick').trim();
    if (!t) return Forge.err('Nothing written.');
    await submitPick(t, true);
  }
  async function submitPick(chosen, custom) {
    const r = await Forge.post(`/api/dpo/${name()}/pick`, {
      prompt: rank.prompt, candidates: rank.candidates, chosen, custom,
    });
    Forge.ok(`Recorded · ${r.pairs_written} pairs written`);
    await drawCandidates();
  }

  // ── misc ───────────────────────────────────────────────
  function newDataset() {
    const box = Forge.modal(`
      <h3>New dataset</h3>
      <p class="hint" style="margin-bottom:12px">Creates an empty <code>datasets/&lt;name&gt;.jsonl</code>.
        Name it after the persona to keep the pipeline simple.</p>
      ${Forge.field('Name', `<input type="text" id="ndName" placeholder="bob">`)}
      <div class="row" style="justify-content:flex-end">
        <button class="btn" data-x="c">Cancel</button>
        <button class="btn primary" data-x="s">Create</button></div>`);
    box.querySelector('[data-x=c]').onclick = Forge.closeModal;
    box.querySelector('[data-x=s]').onclick = Forge.safe(async () => {
      const n = box.querySelector('#ndName').value.trim();
      if (!n) return;
      await Forge.post('/api/dataset/create', { name: n });
      Forge.state.dataset = n;
      Forge.closeModal();
      await Forge.refresh();
      render(document.getElementById('view-data'));
      await paint();
    });
  }

  function importFile() {
    const box = Forge.modal(`
      <h3>Import dataset</h3>
      <p class="hint" style="margin-bottom:12px">Accepts .jsonl, .json or .csv with
        instruction/output (or prompt/response, or chat messages) columns.</p>
      ${Forge.field('Dataset name', `<input type="text" id="imName" value="${Forge.esc(name() || '')}">`)}
      ${Forge.field('File', `<input type="file" id="imFile" accept=".jsonl,.json,.csv">`)}
      <label class="check"><input type="checkbox" id="imReplace"> Replace instead of append</label>
      <div class="row" style="justify-content:flex-end;margin-top:12px">
        <button class="btn" data-x="c">Cancel</button>
        <button class="btn primary" data-x="s">Import</button></div>`);
    box.querySelector('[data-x=c]').onclick = Forge.closeModal;
    box.querySelector('[data-x=s]').onclick = Forge.safe(async () => {
      const f = box.querySelector('#imFile').files[0];
      if (!f) return Forge.err('Pick a file.');
      const fd = new FormData();
      fd.append('name', box.querySelector('#imName').value.trim() || 'imported');
      fd.append('replace', box.querySelector('#imReplace').checked);
      fd.append('file', f);
      const r = await Forge.api('/api/dataset/import', { method: 'POST', body: fd });
      Forge.closeModal();
      Forge.ok(`Imported ${r.imported} rows into ${r.name}`);
      Forge.state.dataset = r.name;
      await Forge.refresh();
      render(document.getElementById('view-data'));
      await paint();
    });
  }

  return {
    render,
    async enter() { await paint(); },
    leave() { clearInterval(genTimer); },
  };
})());
