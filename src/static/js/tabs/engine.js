/* engine.js — start, stop and inspect llama-server; export GGUFs; download
   base models. Everything that used to be a modal with one dropdown. */

Forge.tab('engine', (() => {
  let logTimer = null;

  function render(root) {
    const boot = Forge.state.boot || {};
    const d = boot.defaults?.engine || {};
    const s = Forge.state.engine || {};
    root.innerHTML = `
      <h1 class="page">Engine</h1>
      <p class="page-sub">One llama-server process holds the base model and every character's
        adapter at once, activating whichever one is speaking. Training needs the GPU too —
        stop the engine before a run.</p>

      <div class="grid c2">
        <div class="card">
          <div class="card-head"><h3>Start</h3><div class="spacer"></div>
            <span class="dot ${s.status === 'ready' ? 'ok' : s.status === 'starting' ? 'busy' : ''}"></span>
            <span class="pill">${Forge.esc(s.status || 'stopped')}</span></div>
          ${Forge.field('Base model', Forge.sel('enBase',
            (boot.base_ggufs || []).map(b => [b.name, `${b.name} · ${Forge.bytes(b.bytes)}`]),
            s.base_gguf ? s.base_gguf.split(/[\\/]/).pop() : '')
            .replace('id="__x"', 'id="enBase"'))}
          <div class="grid c3" style="gap:0 12px">
            ${Forge.field('Context', Forge.num('enCtx', d.ctx ?? 8192, { min: 512, step: 512 }), 'per slot')}
            ${Forge.field('GPU layers', Forge.num('enNgl', d.ngl ?? 999, { min: 0 }), '999 = all')}
            ${Forge.field('Slots', Forge.num('enPar', d.parallel ?? 2, { min: 1, max: 8 }),
              '2 = chat + background')}
          </div>
          ${Forge.field('Port', Forge.num('enPort', d.port ?? 8088, { min: 1024, max: 65535 }))}

          <h3 class="sub">Adapters to load</h3>
          <p class="hint" style="margin:0 0 8px">Loaded at launch and switched per message.
            A character whose adapter isn't loaded falls back to the base model.</p>
          <div>${(boot.characters || []).map(c => `
            <label class="check"><input type="checkbox" class="enChar" value="${c.id}"${
              c.adapter_gguf ? ' checked' : ' disabled'}>
              ${Forge.esc(c.name)}${c.adapter_gguf
                ? ` <span class="pill mono">${Forge.esc(c.adapter_gguf.split('/').pop())}</span>`
                : ' <span class="pill">no adapter</span>'}</label>`).join('') ||
            '<p class="hint">No characters yet.</p>'}</div>

          <div class="row" style="margin-top:12px">
            <button class="btn primary" data-act="start">Start engine</button>
            <button class="btn" data-act="stop">Stop</button>
            <div class="spacer"></div>
            <button class="btn sm danger" data-act="kill">Kill orphans</button>
          </div>
          <p class="hint" style="margin-top:8px">llama-server outlives its parent process on
            Windows. "Kill orphans" is the in-app version of
            <code>taskkill /F /IM llama-server.exe /T</code>.</p>
        </div>

        <div class="card">
          <div class="card-head"><h3>Status</h3></div>
          <dl class="kv" id="enKv"></dl>
          <h3 class="sub">Log</h3>
          <div class="log" id="enLog">—</div>
        </div>
      </div>

      <h2 class="sec">GGUF conversion</h2>
      <div class="grid c2">
        <div class="card">
          <div class="card-head"><h3>Export an adapter</h3></div>
          <p class="hint" style="margin-bottom:10px">Converts <code>loras/&lt;name&gt;/</code> into a
            GGUF the engine can load.</p>
          ${Forge.field('Adapter', Forge.sel('gxLora', (boot.loras || []).map(l => l.name))
            .replace('id="__x"', 'id="gxLora"'))}
          ${Forge.field('Precision', Forge.sel('gxType', ['f16', 'f32', 'q8_0'], 'f16')
            .replace('id="__x"', 'id="gxType"'))}
          <button class="btn primary" data-act="export">Export</button>
        </div>
        <div class="card">
          <div class="card-head"><h3>Prepare a base model</h3></div>
          <p class="hint" style="margin-bottom:10px">Converts a downloaded HuggingFace model into
            a quantized base GGUF. This is the 8 GB one, and it takes a while.</p>
          ${Forge.field('Model', Forge.sel('gpModel',
            (boot.models || []).map(m => m.name)).replace('id="__x"', 'id="gpModel"'))}
          ${Forge.field('Quantization', Forge.sel('gpQ', ['q8_0', 'q6_k', 'q5_k_m', 'q4_k_m', 'f16'], 'q8_0')
            .replace('id="__x"', 'id="gpQ"'))}
          <div class="row">
            <button class="btn primary" data-act="prepare">Convert</button>
            <button class="btn sm" data-act="download">Download a model…</button>
          </div>
        </div>
      </div>
      <div class="card"><div class="card-head"><h3>Conversion log</h3></div>
        <div class="log" id="gxLog">Idle.</div></div>

      <h2 class="sec">On disk</h2>
      <div class="grid c2">
        <div class="card"><div class="card-head"><h3>GGUF files</h3></div>
          <div class="tw"><table><thead><tr><th>File</th><th class="num">Size</th></tr></thead><tbody>
            ${(boot.base_ggufs || []).concat(boot.adapters || []).map(g =>
              `<tr><td class="mono">${Forge.esc(g.name)}</td><td class="num">${Forge.bytes(g.bytes)}</td></tr>`
            ).join('') || '<tr><td colspan="2">None</td></tr>'}
          </tbody></table></div></div>
        <div class="card"><div class="card-head"><h3>Trained adapters</h3></div>
          <div class="tw"><table><thead><tr><th>Name</th><th class="num">Rank</th>
            <th class="num">Alpha</th><th class="num">Checkpoints</th><th></th></tr></thead><tbody>
            ${(boot.loras || []).map(l => `<tr><td class="mono">${Forge.esc(l.name)}</td>
              <td class="num">${l.rank ?? '—'}</td><td class="num">${l.alpha ?? '—'}</td>
              <td class="num">${(l.checkpoints || []).length}</td>
              <td><button class="btn sm danger" data-act="delLora" data-n="${Forge.esc(l.name)}">✕</button></td>
            </tr>`).join('') || '<tr><td colspan="5">None</td></tr>'}
          </tbody></table></div></div>
      </div>`;

    Forge.acts(root, {
      start: () => start(),
      stop: Forge.safe(async () => { await Forge.post('/api/engine/stop'); Forge.ok('Engine stopped'); }),
      kill: Forge.safe(async () => {
        if (!await Forge.confirm('Force-kills every llama-server.exe on this machine.',
          { title: 'Kill orphaned engines?', okLabel: 'Kill' })) return;
        const r = await Forge.post('/api/engine/kill_orphans');
        Forge.ok(r.killed ? 'Orphans killed' : 'Nothing was running');
      }),
      export: Forge.safe(async () => {
        await Forge.post('/api/gguf/export',
          { lora_name: Forge.val('gxLora'), outtype: Forge.val('gxType') });
        Forge.ok('Export started'); pollConv();
      }),
      prepare: Forge.safe(async () => {
        if (!await Forge.confirm('Converts and quantizes the full base model. Expect several minutes and ~8 GB of output.',
          { title: 'Convert base model?', danger: false, okLabel: 'Convert' })) return;
        await Forge.post('/api/gguf/prepare_base',
          { model_name: Forge.val('gpModel'), quantize: Forge.val('gpQ') });
        Forge.ok('Conversion started'); pollConv();
      }),
      download: () => downloadDialog(),
      delLora: Forge.safe(async (el) => {
        if (!await Forge.confirm(`Deletes loras/${el.dataset.n}/ and its checkpoints. The exported GGUF stays.`,
          { title: 'Delete adapter?', okLabel: 'Delete' })) return;
        await Forge.del(`/api/loras/${el.dataset.n}`);
        await Forge.refresh();
        render(root);
      }),
    });
    paintStatus(Forge.state.engine);
  }

  async function start() {
    const base = Forge.val('enBase');
    if (!base) return Forge.err('No base GGUF available — convert one below first.');
    const ids = Forge.$$('.enChar').filter(c => c.checked && !c.disabled).map(c => c.value);
    await Forge.post('/api/engine/start', {
      base_gguf: base,
      char_ids: ids,
      ctx: Forge.numv('enCtx', 8192),
      ngl: Forge.numv('enNgl', 999),
      parallel: Forge.numv('enPar', 2),
      port: Forge.numv('enPort', 8088),
    });
    Forge.ok('Engine starting — this takes a moment while the model loads');
  }

  function paintStatus(s) {
    const kv = document.getElementById('enKv');
    if (!kv || !s) return;
    kv.innerHTML = `
      <dt>Status</dt><dd>${Forge.esc(s.status || 'stopped')}</dd>
      <dt>Base</dt><dd>${Forge.esc((s.base_gguf || '—').split(/[\\/]/).pop())}</dd>
      <dt>Port</dt><dd>${s.port ?? '—'}</dd>
      <dt>Context</dt><dd>${s.ctx ?? '—'} × ${s.parallel ?? 1} slots</dd>
      <dt>Adapters</dt><dd>${(s.adapters || []).map(a => a.name).join(', ') || 'none'}</dd>
      ${s.error ? `<dt>Error</dt><dd style="color:var(--crit)">${Forge.esc(s.error)}</dd>` : ''}`;
    const log = document.getElementById('enLog');
    if (log && s.log) {
      const at = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
      log.textContent = s.log.slice(-120).join('\n');
      if (at) log.scrollTop = log.scrollHeight;
    }
  }

  function pollConv() {
    clearInterval(logTimer);
    logTimer = setInterval(async () => {
      const box = document.getElementById('gxLog');
      if (!box) return clearInterval(logTimer);
      const s = await Forge.get('/api/gguf/status').catch(() => null);
      if (!s) return;
      box.textContent = (s.log || []).slice(-80).join('\n') || 'Idle.';
      box.scrollTop = box.scrollHeight;
      if (!s.running) {
        clearInterval(logTimer);
        if (s.status === 'done') { await Forge.refresh(); Forge.ok('Conversion finished'); }
      }
    }, 2500);
  }

  function downloadDialog() {
    const box = Forge.modal(`
      <h3>Download a base model</h3>
      <p class="hint" style="margin-bottom:12px">Pulls from HuggingFace into <code>models/</code>.
        Qwen3-8B is about 16 GB. You still need to convert it to GGUF afterwards.</p>
      ${Forge.field('Model id', `<input type="text" id="dlName" class="mono" value="Qwen/Qwen3-8B">`)}
      <div class="row" style="justify-content:flex-end">
        <button class="btn" data-x="c">Cancel</button>
        <button class="btn primary" data-x="s">Download</button></div>
      <div class="log" id="dlLog" style="margin-top:12px" hidden></div>`);
    box.querySelector('[data-x=c]').onclick = Forge.closeModal;
    box.querySelector('[data-x=s]').onclick = Forge.safe(async () => {
      await Forge.post('/api/model/download', { model_name: box.querySelector('#dlName').value.trim() });
      const log = box.querySelector('#dlLog');
      log.hidden = false;
      const t = setInterval(async () => {
        const s = await Forge.get('/api/model/status').catch(() => null);
        if (!s) return;
        log.textContent = (s.log || []).slice(-40).join('\n');
        log.scrollTop = log.scrollHeight;
        if (!s.running) { clearInterval(t); await Forge.refresh(); }
      }, 2000);
    });
  }

  return {
    render,
    onStatus(_tr, eng) { paintStatus(eng); },
    leave() { clearInterval(logTimer); },
  };
})());
