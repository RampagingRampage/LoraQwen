/* evaluate.js — the Evaluate tab.
   Side-by-side generation, the style scorecard, the memorization check and the
   regression set. Held-out loss lives in the Train tab's curve; this is
   everything you can only see by actually generating. */

Forge.tab('evaluate', (() => {
  let mode = 'compare';

  function render(root) {
    const boot = Forge.state.boot || {};
    root.innerHTML = `
      <h1 class="page">Evaluate</h1>
      <p class="page-sub">Did the training take? Train loss going down doesn't answer that.
        These four checks do — comparison against the base model, measurable style match,
        memorization risk, and drift between runs.</p>

      <div class="row" style="margin-bottom:16px">
        ${[['compare', 'Side by side'], ['style', 'Style scorecard'],
           ['memo', 'Memorization'], ['regression', 'Regression set']].map(([m, l]) =>
          `<button class="btn sm${m === mode ? ' primary' : ''}" data-act="mode" data-m="${m}">${l}</button>`).join('')}
      </div>
      <div id="evalBody"></div>`;

    Forge.acts(root, {
      mode: (el) => { mode = el.dataset.m; render(root); },
      compare: () => runCompare(),
      style: () => runStyle(),
      memo: () => runMemo(),
      runReg: () => runRegression(),
      saveReg: () => saveRegression(),
    });
    paint();
  }

  const chars = () => Forge.state.boot?.characters || [];
  const dsets = () => Forge.state.boot?.datasets || [];

  function paint() {
    const body = document.getElementById('evalBody');
    if (!body) return;
    if (!chars().length) {
      body.innerHTML = `<div class="empty"><div class="big">◎</div>
        <p><strong>Nothing to evaluate yet.</strong></p>
        <p>Train an adapter and register it as a character first.</p></div>`;
      return;
    }
    if (mode === 'compare') return paintCompare(body);
    if (mode === 'style') return paintStyle(body);
    if (mode === 'memo') return paintMemo(body);
    if (mode === 'regression') return paintRegression(body);
  }

  // ── COMPARE ────────────────────────────────────────────
  function paintCompare(body) {
    body.innerHTML = `
      <div class="card">
        <div class="card-head"><h3>One prompt, several adapters</h3></div>
        <p class="hint" style="margin-bottom:12px">
          Only adapters loaded into the running engine can answer — restart the
          engine to include one that isn't. This is what the old
          <code>*_pre_dpo_backup</code> snapshots should have been for.</p>
        ${Forge.field('Prompt', `<textarea id="cmpPrompt" rows="2"
          placeholder="whats your take on people using ai to write all their code"></textarea>`)}
        <h3 class="sub">Compare</h3>
        <div class="row" style="margin-bottom:10px">
          ${chars().map(c => `<label class="check"><input type="checkbox" class="cmpChar"
            value="${c.id}" checked> ${Forge.esc(c.name)}</label>`).join('')}
          <label class="check"><input type="checkbox" id="cmpBase" checked> base model</label>
          <label class="check"><input type="checkbox" id="cmpPersona" checked> use persona prompt</label>
        </div>
        <div class="row">
          ${Forge.field('Temperature', Forge.num('cmpTemp', 0.8, { min: 0, max: 2, step: 0.05 }))}
          ${Forge.field('Max tokens', Forge.num('cmpTok', 200, { min: 32, max: 2048, step: 32 }))}
          <button class="btn primary" data-act="compare" style="margin-top:6px">Generate</button>
        </div>
      </div>
      <div id="cmpOut"></div>`;
  }

  async function runCompare() {
    const prompt = Forge.val('cmpPrompt').trim();
    if (!prompt) return Forge.err('Type a prompt first.');
    const ids = Forge.$$('.cmpChar').filter(c => c.checked).map(c => c.value);
    document.getElementById('cmpOut').innerHTML = '<p class="hint">Generating…</p>';
    const r = await Forge.post('/api/eval/compare', {
      prompt, char_ids: ids,
      include_base: Forge.chk('cmpBase'),
      use_persona: Forge.chk('cmpPersona'),
      temperature: Forge.numv('cmpTemp', 0.8),
      max_tokens: Forge.numv('cmpTok', 200),
    });
    document.getElementById('cmpOut').innerHTML = `<div class="split-scroll">${
      r.results.map(x => `
        <div class="card">
          <div class="card-head"><h3>${Forge.esc(x.label)}</h3><div class="spacer"></div>
            ${x.adapter_loaded ? '' : '<span class="pill crit">not loaded</span>'}
            <span class="pill">${x.seconds}s</span></div>
          <div style="font-size:13.5px;white-space:pre-wrap;line-height:1.6">${
            Forge.esc(x.text || x.error || '(nothing)')}</div>
          ${x.style ? `<div class="row tight" style="margin-top:10px">
            <span class="pill">${Math.round(x.style.lowercase_ratio * 100)}% lowercase</span>
            <span class="pill">${x.style.avg_sentence_words.toFixed(1)} w/sentence</span>
            <span class="pill">${x.style.words} words</span></div>` : ''}
        </div>`).join('')}</div>`;
  }

  // ── STYLE ──────────────────────────────────────────────
  function paintStyle(body) {
    body.innerHTML = `
      <div class="card">
        <div class="card-head"><h3>Does it sound like the training data?</h3></div>
        <p class="hint" style="margin-bottom:12px">
          Generates against the regression prompts, then measures the output against
          the dataset's own measurable style. A persona specified as "lowercase, drops
          apostrophes, sparse punctuation" is directly checkable — this checks it.</p>
        <div class="grid c3" style="gap:0 12px">
          ${Forge.field('Character', Forge.sel('stChar', chars().map(c => [c.id, c.name]),
            Forge.state.char).replace('id="__x"', 'id="stChar"'))}
          ${Forge.field('Compare against dataset', Forge.sel('stData', dsets().map(d => d.name),
            Forge.state.dataset).replace('id="__x"', 'id="stData"'))}
          ${Forge.field('Prompts to use', Forge.num('stN', 6, { min: 2, max: 30 }))}
        </div>
        <button class="btn primary" data-act="style">Run scorecard</button>
      </div>
      <div id="stOut"></div>`;
  }

  async function runStyle() {
    document.getElementById('stOut').innerHTML = '<p class="hint">Generating and measuring…</p>';
    const r = await Forge.post('/api/eval/style', {
      char_id: Forge.val('stChar'), dataset: Forge.val('stData'), n: Forge.numv('stN', 6),
    });
    const verdictPill = v => `<span class="pill ${v === 'close' ? 'ok' : v === 'drifting' ? 'warn' : 'crit'}">${v}</span>`;
    document.getElementById('stOut').innerHTML = `
      <div class="card">
        <div class="card-head"><h3>${Forge.esc(r.character)} vs ${Forge.esc(r.dataset)}</h3>
          <div class="spacer"></div>
          <span class="pill ${r.overall_distance < 0.25 ? 'ok' : r.overall_distance < 0.45 ? 'warn' : 'crit'}">
            overall distance ${r.overall_distance}</span></div>
        <div class="tw"><table><thead><tr><th>Dimension</th><th class="num">Dataset</th>
          <th class="num">Model</th><th class="num">Gap</th><th></th></tr></thead><tbody>
          ${r.dimensions.map(d => `<tr><td>${Forge.esc(d.label)}</td>
            <td class="num">${d.dataset}</td><td class="num">${d.model}</td>
            <td class="num">${d.gap}</td><td>${verdictPill(d.verdict)}</td></tr>`).join('')}
        </tbody></table></div>
      </div>
      <div class="card">
        <div class="card-head"><h3>What it actually said</h3></div>
        ${r.samples.map(s => `<div style="margin-bottom:14px">
          <div class="hint" style="margin:0 0 3px">${Forge.esc(s.prompt)}</div>
          <div style="font-size:13px;white-space:pre-wrap">${Forge.esc(s.text || s.error || '')}</div>
          ${s.distance != null ? `<span class="pill ${s.distance < 0.3 ? 'ok' : 'warn'}"
            style="margin-top:4px;display:inline-block">distance ${s.distance}</span>` : ''}
        </div>`).join('')}
      </div>`;
  }

  // ── MEMORIZATION ───────────────────────────────────────
  function paintMemo(body) {
    body.innerHTML = `
      <div class="card">
        <div class="card-head"><h3>Is it reciting its training data?</h3></div>
        <p class="hint" style="margin-bottom:12px">
          Answers prompts it was trained on, then finds the closest training row to
          each answer. Some overlap is expected — most of them near-verbatim means
          the adapter memorized rather than generalized, which is the real risk at
          a few hundred examples.</p>
        <div class="grid c3" style="gap:0 12px">
          ${Forge.field('Character', Forge.sel('mmChar', chars().map(c => [c.id, c.name]),
            Forge.state.char).replace('id="__x"', 'id="mmChar"'))}
          ${Forge.field('Dataset', Forge.sel('mmData', dsets().map(d => d.name),
            Forge.state.dataset).replace('id="__x"', 'id="mmData"'))}
          ${Forge.field('Similarity threshold', Forge.num('mmThr', 0.82, { min: 0.5, max: 0.99, step: 0.01 }))}
        </div>
        ${Forge.field('Prompts to check', Forge.num('mmN', 12, { min: 3, max: 40 }))}
        <button class="btn primary" data-act="memo">Run check</button>
      </div>
      <div id="mmOut"></div>`;
  }

  async function runMemo() {
    document.getElementById('mmOut').innerHTML = '<p class="hint">Generating and comparing…</p>';
    const r = await Forge.post('/api/eval/memorization', {
      char_id: Forge.val('mmChar'), dataset: Forge.val('mmData'),
      threshold: Forge.numv('mmThr', 0.82), n: Forge.numv('mmN', 12),
    });
    const kind = r.memorized_pct > 50 ? 'crit' : r.memorized_pct > 20 ? 'warn' : 'ok';
    document.getElementById('mmOut').innerHTML = `
      <div class="grid c4" style="margin-bottom:14px">
        <div class="stat"><div class="k">Checked</div><div class="v">${r.checked}</div></div>
        <div class="stat"><div class="k">Memorized</div>
          <div class="v" style="color:var(--${kind})">${r.memorized}</div>
          <div class="n">${r.memorized_pct}% of prompts</div></div>
        <div class="stat"><div class="k">Mean similarity</div><div class="v">${r.mean_similarity}</div></div>
        <div class="stat"><div class="k">Verdict</div><div class="v sm" style="color:var(--${kind})">${
          r.memorized_pct > 50 ? 'overfit' : r.memorized_pct > 20 ? 'watch it' : 'healthy'}</div></div>
      </div>
      ${r.findings.map(f => `
        <div class="card tight">
          <div class="card-head">
            <span class="pill ${f.memorized ? 'crit' : 'ok'}">${f.similarity}</span>
            <span class="hint" style="margin:0">${Forge.esc(f.prompt)}</span></div>
          <div class="grid c2">
            <div><div class="hint">generated</div>
              <div style="font-size:13px;white-space:pre-wrap">${Forge.esc(f.generated)}</div></div>
            <div><div class="hint">closest training row</div>
              <div style="font-size:13px;white-space:pre-wrap;color:var(--dim)">${
                Forge.esc(f.closest_training_row)}</div></div>
          </div>
        </div>`).join('')}`;
  }

  // ── REGRESSION ─────────────────────────────────────────
  async function paintRegression(body) {
    const p = await Forge.get('/api/eval/regression/prompts');
    body.innerHTML = `
      <div class="card">
        <div class="card-head"><h3>Pinned prompts</h3><div class="spacer"></div>
          <button class="btn sm" data-act="saveReg">Save list</button></div>
        <p class="hint" style="margin-bottom:10px">Re-run these after every training pass.
          Each run is diffed against the previous one, so you can see what a change
          actually changed.</p>
        <textarea id="regPrompts" class="mono" rows="9">${Forge.esc(p.prompts.join('\n'))}</textarea>
      </div>
      <div class="card">
        <div class="card-head"><h3>Run</h3><div class="spacer"></div>
          ${Forge.sel('regChar', chars().map(c => [c.id, c.name]), Forge.state.char)
            .replace('id="__x"', 'id="regChar" style="max-width:180px"')}
          <button class="btn primary sm" data-act="runReg">Run regression</button></div>
        <div id="regOut"></div>
      </div>`;
    loadRegHistory();
  }

  async function saveRegression() {
    const prompts = Forge.val('regPrompts').split('\n').map(s => s.trim()).filter(Boolean);
    await Forge.post('/api/eval/regression/prompts', { prompts });
    Forge.ok(`${prompts.length} prompts saved`);
  }

  async function runRegression() {
    document.getElementById('regOut').innerHTML = '<p class="hint">Running every prompt…</p>';
    const r = await Forge.post('/api/eval/regression/run', { char_id: Forge.val('regChar') });
    paintRegRun(r);
  }

  async function loadRegHistory() {
    const cid = Forge.val('regChar') || Forge.state.char;
    if (!cid) return;
    const runs = await Forge.get(`/api/eval/regression/${cid}`).catch(() => []);
    if (runs.length) paintRegRun(runs[runs.length - 1], runs.length);
  }

  function paintRegRun(r, count) {
    const changed = r.results.filter(x => x.changed).length;
    document.getElementById('regOut').innerHTML = `
      <div class="row" style="margin:10px 0">
        <span class="pill">${Forge.esc(r.label)}</span>
        <span class="pill ${changed ? 'warn' : 'ok'}">${changed} of ${r.results.length} changed</span>
        ${count ? `<span class="pill">run ${count}</span>` : ''}
      </div>
      ${r.results.map(x => `
        <div style="border-top:1px solid var(--line-2);padding:10px 0">
          <div class="hint" style="margin:0 0 4px">${Forge.esc(x.prompt)}</div>
          <div style="font-size:13px;white-space:pre-wrap">${Forge.esc(x.text || x.error || '')}</div>
          ${x.previous ? `<details style="margin-top:6px">
            <summary class="hint" style="cursor:pointer">previous${
              x.drift != null ? ` · drift ${x.drift}` : ''}</summary>
            <div style="font-size:12.5px;color:var(--dim);white-space:pre-wrap;margin-top:4px">${
              Forge.esc(x.previous)}</div></details>` : ''}
        </div>`).join('')}`;
  }

  return { render, enter() { paint(); } };
})());
