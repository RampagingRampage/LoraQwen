/* persona.js — the Persona tab.
   Upload raw documents, preview and apply the filter, run distillation with
   live progress, and hand-edit the resulting profile. */

Forge.tab('persona', (() => {
  let data = null;
  let timer = null;
  const name = () => Forge.state.persona;

  function render(root) {
    const personas = Forge.state.boot?.personas || [];
    root.innerHTML = `
      <h1 class="page">Persona</h1>
      <p class="page-sub">Turn raw writing — chat logs, posts, anything in your own words —
        into a structured voice profile. The model reads it in ~1,400-character chunks and
        merges what it learns, chunk by chunk.</p>

      <div class="row" style="margin-bottom:16px">
        <select id="pSel" style="max-width:220px">
          ${personas.map(p => `<option value="${p.name}"${p.name === name() ? ' selected' : ''}>${
            Forge.esc(p.name)}${p.has_master ? ' ✓' : ''}</option>`).join('') || '<option value="">(none)</option>'}
        </select>
        <button class="btn sm" data-act="newp">New persona</button>
        <div class="spacer"></div>
        <button class="btn sm" data-act="upload">Add documents</button>
      </div>

      <div id="pBody"></div>`;

    Forge.acts(root, {
      newp: () => newPersona(),
      upload: () => upload(),
      filter: (el) => runFilter(el.dataset.f, false),
      applyFilter: (el) => runFilter(el.dataset.f, true),
      build: () => build(),
      saveMd: Forge.safe(async () => {
        await Forge.put(`/api/persona/${name()}`, { persona_md: Forge.val('pMd') });
        Forge.ok('persona.md saved');
      }),
      saveMaster: Forge.safe(async () => {
        let parsed;
        try { parsed = JSON.parse(Forge.val('pMaster')); }
        catch (e) { return Forge.err('That is not valid JSON: ' + e.message); }
        await Forge.put(`/api/persona/${name()}`, { master: parsed });
        Forge.ok('Profile saved');
        await load();
      }),
      toData: () => { Forge.state.dataset = name(); Forge.go('data'); },
    });

    document.getElementById('pSel').onchange = Forge.safe(async (e) => {
      Forge.state.persona = e.target.value;
      await load();
    });
  }

  async function load() {
    const box = document.getElementById('pBody');
    if (!box) return;
    if (!name()) {
      box.innerHTML = `<div class="empty"><div class="big">◇</div>
        <p><strong>No persona yet.</strong></p>
        <p>Create one, then drop in your writing — chat exports, posts, anything
           unedited and in your own voice. Thousands of lines beats a curated few hundred.</p></div>`;
      return;
    }
    data = await Forge.get(`/api/persona/${name()}`);
    paint();
  }

  function paint() {
    const box = document.getElementById('pBody');
    const raw = data.documents.filter(d => /\.(txt|json|jsonl|md|csv)$/i.test(d.file)
      && !['master_persona.json', 'persona.md'].includes(d.file));
    box.innerHTML = `
      <div class="grid c2">
        <div class="card">
          <div class="card-head"><h3>Source documents</h3><div class="spacer"></div>
            <span class="pill">${raw.length} files</span></div>
          ${raw.length ? `<div class="tw"><table><thead><tr><th>File</th><th class="num">Size</th>
            <th></th></tr></thead><tbody>${raw.map(d => `<tr>
              <td class="mono">${Forge.esc(d.file)}</td>
              <td class="num">${Forge.bytes(d.bytes)}</td>
              <td><div class="row tight">
                <button class="btn sm" data-act="filter" data-f="${Forge.esc(d.file)}">Preview filter</button>
              </div></td></tr>`).join('')}</tbody></table></div>`
            : `<p class="hint">Nothing here yet — click "Add documents".</p>`}
          <div id="filterOut"></div>
        </div>

        <div class="card">
          <div class="card-head"><h3>Distill</h3></div>
          <p class="hint" style="margin-bottom:12px">Reads every document and builds the
            profile. Needs the engine running. A big corpus takes a while — the sample
            option reads a random spread instead of everything, which is usually enough.</p>
          <div class="grid c2" style="gap:0 12px">
            ${Forge.field('Sample N chunks', Forge.num('pSample', 200, { min: 0 }), '0 = read everything')}
            ${Forge.field('Hard limit', Forge.num('pLimit', 0, { min: 0 }), '0 = no limit')}
          </div>
          <button class="btn primary" data-act="build">Run distillation</button>
          <div id="buildBox" style="margin-top:12px"></div>
        </div>
      </div>

      <h2 class="sec">persona.md — the short prompt version</h2>
      <div class="card">
        <textarea id="pMd" rows="7">${Forge.esc(data.persona_md || '')}</textarea>
        <div class="row" style="margin-top:8px">
          <button class="btn sm" data-act="saveMd">Save</button>
          ${data.persona_md ? `<button class="btn sm" data-act="toData">Generate data from this →</button>` : ''}
        </div>
      </div>

      <h2 class="sec">Voice profile</h2>
      ${data.topics?.length ? `<div class="row" style="margin-bottom:10px">
        <span class="hint" style="margin:0">Topic pool (${data.topics.length}):</span>
        ${data.topics.slice(0, 18).map(t => `<span class="pill">${Forge.esc(t)}</span>`).join('')}
        ${data.topics.length > 18 ? `<span class="pill">+${data.topics.length - 18} more</span>` : ''}
      </div>` : ''}
      <div class="card">
        <textarea id="pMaster" class="mono" rows="16">${
          Forge.esc(JSON.stringify(data.master || {}, null, 2))}</textarea>
        <div class="row" style="margin-top:8px">
          <button class="btn sm" data-act="saveMaster">Save profile</button>
          <span class="hint" style="margin:0">Edit the numbers and lists directly —
            the generator reads this, so a wrong trait here shows up in every sample.</span>
        </div>
      </div>`;
  }

  async function runFilter(file, apply) {
    const r = await Forge.post(`/api/persona/${name()}/filter`, { file, apply, min_chars: 12 });
    document.getElementById('filterOut').innerHTML = `
      <h3 class="sub">${Forge.esc(file)}</h3>
      <div class="grid c3" style="margin-bottom:10px">
        <div class="stat"><div class="k">Kept</div><div class="v" style="color:var(--ok)">${r.kept}</div></div>
        <div class="stat"><div class="k">Dropped</div><div class="v" style="color:var(--warn)">${r.dropped}</div></div>
        <div class="stat"><div class="k">Total lines</div><div class="v sm">${r.total}</div></div>
      </div>
      ${r.written ? `<p class="hint" style="color:var(--ok)">Wrote ${Forge.esc(r.written)}</p>`
        : `<button class="btn sm" data-act="applyFilter" data-f="${Forge.esc(file)}">Write the filtered file</button>`}
      ${r.dropped_samples?.length ? `<h3 class="sub">Examples of what gets dropped</h3>
        <div class="log">${r.dropped_samples.map(Forge.esc).join('\n')}</div>` : ''}`;
  }

  async function build() {
    if (!await Forge.confirm('Reads every document through the model. This can take a long time on a big corpus.',
      { title: 'Run distillation?', danger: false, okLabel: 'Run' })) return;
    await Forge.post(`/api/persona/${name()}/build`, {
      sample: Forge.numv('pSample', 0) || undefined,
      limit: Forge.numv('pLimit', 0) || undefined,
    });
    Forge.ok('Distillation started');
    poll();
  }

  function poll() {
    clearInterval(timer);
    timer = setInterval(async () => {
      const box = document.getElementById('buildBox');
      if (!box) return clearInterval(timer);
      const s = await Forge.get('/api/persona/build_status').catch(() => null);
      if (!s) return;
      box.innerHTML = `
        <div class="row" style="margin-bottom:6px">
          <span class="pill ${s.status === 'error' ? 'crit' : s.status === 'done' ? 'ok' : 'accent'}">${
            Forge.esc(s.status)}</span>
          ${s.total ? `<span class="pill">${s.done}/${s.total} chunks</span>` : ''}
        </div>
        ${s.total ? `<div class="bar"><i style="width:${Math.round(s.done / s.total * 100)}%"></i></div>` : ''}
        ${s.error ? `<p class="hint" style="color:var(--crit)">${Forge.esc(s.error)}</p>` : ''}`;
      if (!s.running) {
        clearInterval(timer);
        if (s.status === 'done') { await Forge.refresh(); await load(); Forge.ok('Profile built'); }
      }
    }, 2500);
  }

  function newPersona() {
    const box = Forge.modal(`
      <h3>New persona</h3>
      <p class="hint" style="margin-bottom:12px">Creates <code>training_input/&lt;name&gt;/</code>.
        Use the same name for the dataset and the adapter to keep the pipeline simple.</p>
      ${Forge.field('Name', `<input type="text" id="npName" placeholder="bob">`)}
      <div class="row" style="justify-content:flex-end">
        <button class="btn" data-x="c">Cancel</button>
        <button class="btn primary" data-x="s">Create</button></div>`);
    box.querySelector('[data-x=c]').onclick = Forge.closeModal;
    box.querySelector('[data-x=s]').onclick = Forge.safe(async () => {
      const n = box.querySelector('#npName').value.trim();
      if (!n) return;
      await Forge.put(`/api/persona/${n}`, { persona_md: '' });
      Forge.state.persona = n;
      Forge.closeModal();
      await Forge.refresh();
      render(document.getElementById('view-persona'));
      await load();
    });
  }

  function upload() {
    const box = Forge.modal(`
      <h3>Add documents to "${Forge.esc(name())}"</h3>
      <p class="hint" style="margin-bottom:12px">.txt, .json, .jsonl, .md or .csv.
        Raw and unedited is better than tidy — the distiller wants how you actually write.</p>
      ${Forge.field('Files', `<input type="file" id="upDocs" multiple accept=".txt,.json,.jsonl,.md,.csv">`)}
      <div class="row" style="justify-content:flex-end">
        <button class="btn" data-x="c">Cancel</button>
        <button class="btn primary" data-x="s">Upload</button></div>`);
    box.querySelector('[data-x=c]').onclick = Forge.closeModal;
    box.querySelector('[data-x=s]').onclick = Forge.safe(async () => {
      const files = [...box.querySelector('#upDocs').files];
      if (!files.length) return;
      const fd = new FormData();
      files.forEach(f => fd.append('files', f, f.name));
      const r = await Forge.api(`/api/persona/${name()}/upload`, { method: 'POST', body: fd });
      Forge.closeModal();
      Forge.ok(`${r.saved.length} file(s) added`);
      await load();
    });
  }

  return {
    render,
    async enter() { await load(); },
    leave() { clearInterval(timer); },
  };
})());
