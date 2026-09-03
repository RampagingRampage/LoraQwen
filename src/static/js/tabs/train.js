/* train.js — the Train tab.
   Every hyperparameter exposed, three presets, and a live train/eval loss
   curve drawn on a canvas (no charting library — it's two polylines). */

Forge.tab('train', (() => {
  let history = [];
  let lastStatus = null;
  let exportTimer = null;

  function d() { return Forge.state.boot?.train_defaults || {}; }

  function render(root) {
    const boot = Forge.state.boot || {};
    const presets = boot.train_presets || {};
    const t = d();
    root.innerHTML = `
      <h1 class="page">Train</h1>
      <p class="page-sub">QLoRA over a 4-bit base — about 0.3% of the weights are trainable.
        Training runs in a child process so every byte of VRAM comes back when it exits.
        Watch the loss curve; watch the eval curve more closely.</p>

      <div class="grid c2">
        <div class="card">
          <div class="card-head"><h3>What to train</h3></div>
          ${Forge.field('Adapter name', `<input type="text" id="trName" value="${
            Forge.esc(Forge.state.dataset || '')}">`, 'writes to loras/<name>/')}
          ${Forge.field('Dataset', Forge.sel('trDataset',
            (boot.datasets || []).map(x => [x.name, `${x.name} · ${x.rows} rows`]),
            Forge.state.dataset).replace('id="__x"', 'id="trDataset"'))}
          ${Forge.field('Base model', Forge.sel('trModel',
            (boot.models || []).map(m => m.name).concat(
              (boot.models || []).some(m => m.name === t.model_name) ? [] : [t.model_name]),
            t.model_name).replace('id="__x"', 'id="trModel"'),
            'downloaded models plus the default')}

          <h3 class="sub">Presets</h3>
          <div class="row tight">
            ${Object.entries(presets).map(([k, p]) =>
              `<button class="btn sm" data-act="preset" data-p="${k}" title="${Forge.esc(p.note)}">${
                Forge.esc(p.label)}</button>`).join('')}
          </div>
          <p class="hint" id="presetNote">Standard is the known-good baseline.</p>
        </div>

        <div class="card">
          <div class="card-head"><h3>Hyperparameters</h3></div>
          <div class="grid c2" style="gap:0 12px">
            ${Forge.field('LoRA rank', Forge.num('trRank', t.lora_rank, { min: 2, max: 256 }),
              'capacity')}
            ${Forge.field('LoRA alpha', Forge.num('trAlpha', t.lora_alpha, { min: 2, max: 512 }),
              'usually 2× rank')}
            ${Forge.field('Dropout', Forge.num('trDropout', t.lora_dropout, { min: 0, max: 0.5, step: 0.01 }))}
            ${Forge.field('Sequence length', Forge.num('trSeq', t.max_seq_length, { min: 128, step: 128 }),
              'longer = more VRAM')}
            ${Forge.field('Epochs', Forge.num('trEpochs', t.num_epochs, { min: 1, max: 30 }))}
            ${Forge.field('Batch size', Forge.num('trBatch', t.batch_size, { min: 1, max: 32 }))}
            ${Forge.field('Grad accumulation', Forge.num('trAccum', t.grad_accum, { min: 1, max: 64 }))}
            ${Forge.field('Learning rate', Forge.num('trLr', t.learning_rate, { min: 0.000001, step: 0.00001 }))}
            ${Forge.field('Warmup ratio', Forge.num('trWarmup', t.warmup_ratio, { min: 0, max: 0.5, step: 0.01 }))}
            ${Forge.field('Scheduler', Forge.sel('trSched',
              ['cosine', 'linear', 'constant', 'constant_with_warmup'], t.lr_scheduler)
              .replace('id="__x"', 'id="trSched"'))}
            ${Forge.field('Eval split %', Forge.num('trEval', t.eval_split_pct, { min: 0, max: 50 }),
              '0 = no eval curve')}
            ${Forge.field('Seed', Forge.num('trSeed', t.seed, { min: 0 }))}
            ${Forge.field('Log every N steps', Forge.num('trLog', t.logging_steps, { min: 1, max: 100 }))}
            ${Forge.field('Save every N steps', Forge.num('trSave', t.save_steps, { min: 10 }))}
          </div>
          ${Forge.field('Target modules', `<input type="text" id="trTargets" class="mono" value="${
            Forge.esc(t.target_modules)}">`, 'add gate_proj,up_proj,down_proj for more capacity')}
          <p class="hint" id="vramHint"></p>
        </div>
      </div>

      <div class="row" style="margin:4px 0 16px">
        <button class="btn primary" data-act="start">Start training</button>
        <button class="btn danger" data-act="stop">Stop (saves the adapter)</button>
        <div class="spacer"></div>
        <button class="btn sm" data-act="export">Export GGUF</button>
        <button class="btn sm" data-act="register">Register as character</button>
      </div>

      <div class="card">
        <div class="card-head"><h3>Live</h3><div class="spacer"></div>
          <span class="pill" id="trStatus">idle</span></div>
        <div class="grid c4" style="margin-bottom:14px" id="trStats"></div>
        <div class="bar"><i id="trBar" style="width:0%"></i></div>
        <div class="chart" style="margin-top:16px"><canvas id="lossChart"></canvas></div>
        <div class="legend">
          <span><i style="background:var(--accent)"></i>train loss</span>
          <span><i style="background:var(--warn)"></i>eval loss</span>
          <span id="lossNote" style="margin-left:auto;color:var(--faint)"></span>
        </div>
        <div class="log" id="trLog" style="margin-top:14px">Not running.</div>
      </div>

      <h2 class="sec">Previous runs</h2>
      <div id="runsBox"></div>`;

    Forge.acts(root, {
      preset: (el) => applyPreset(el.dataset.p),
      start: () => start(),
      stop: Forge.safe(async () => {
        await Forge.post('/api/train/stop');
        Forge.ok('Stopping — the adapter is still saved');
      }),
      export: Forge.safe(async () => {
        await Forge.post('/api/gguf/export', { lora_name: Forge.val('trName') });
        Forge.ok('Export started…');
        pollExport();
      }),
      register: () => registerChar(),
    });

    ['trRank', 'trSeq', 'trBatch'].forEach(id => {
      const e = document.getElementById(id);
      if (e) e.addEventListener('input', vramHint);
    });
    vramHint();
    loadRuns();
  }

  function applyPreset(key) {
    const p = (Forge.state.boot.train_presets || {})[key];
    if (!p) return;
    document.getElementById('trRank').value = p.lora_rank;
    document.getElementById('trAlpha').value = p.lora_alpha;
    document.getElementById('trEpochs').value = p.num_epochs;
    document.getElementById('presetNote').textContent = p.note;
    vramHint();
  }

  function vramHint() {
    // Deliberately rough. The point is to catch "rank 64 at seq 4096 batch 8"
    // before it OOMs 40 seconds in, not to be accurate to the megabyte.
    const rank = Forge.numv('trRank', 16);
    const seq = Forge.numv('trSeq', 1024);
    const batch = Forge.numv('trBatch', 2);
    const est = 6.2 + (seq / 1024) * batch * 1.6 + (rank / 16) * 0.35;
    const el = document.getElementById('vramHint');
    if (!el) return;
    el.innerHTML = `Rough VRAM estimate: <strong>~${est.toFixed(1)} GB</strong>` +
      (est > 11 ? ' <span class="pill crit">tight on a 12 GB card</span>' : '');
  }

  async function start() {
    const name = Forge.val('trName').trim();
    if (!name) return Forge.err('Give the adapter a name.');
    const cfg = {
      lora_name: name,
      dataset: Forge.val('trDataset'),
      model_name: Forge.val('trModel'),
      lora_rank: Forge.numv('trRank', 16),
      lora_alpha: Forge.numv('trAlpha', 32),
      lora_dropout: Forge.numv('trDropout', 0.05),
      max_seq_length: Forge.numv('trSeq', 1024),
      num_epochs: Forge.numv('trEpochs', 3),
      batch_size: Forge.numv('trBatch', 2),
      grad_accum: Forge.numv('trAccum', 4),
      learning_rate: Forge.numv('trLr', 0.0002),
      warmup_ratio: Forge.numv('trWarmup', 0.03),
      lr_scheduler: Forge.val('trSched'),
      eval_split_pct: Forge.numv('trEval', 10),
      seed: Forge.numv('trSeed', 42),
      logging_steps: Forge.numv('trLog', 5),
      save_steps: Forge.numv('trSave', 200),
      target_modules: Forge.val('trTargets'),
    };
    if (!await Forge.confirm(
      `Trains "${name}" on ${cfg.dataset} for ${cfg.num_epochs} epochs at rank ${cfg.lora_rank}. ` +
      `This takes over the GPU — stop the engine first if it's running.`,
      { title: 'Start training?', danger: false, okLabel: 'Start' })) return;
    await Forge.post('/api/train/start', cfg);
    Forge.ok('Training started');
  }

  function registerChar() {
    const name = Forge.val('trName').trim();
    const box = Forge.modal(`
      <h3>Register "${Forge.esc(name)}" as a character</h3>
      <p class="hint" style="margin-bottom:12px">Points a character at the exported adapter GGUF
        so the engine can load it and you can chat with it.</p>
      ${Forge.field('Adapter GGUF', Forge.sel('regAdapter',
        (Forge.state.boot.adapters || []).map(a => a.name), `${name}-adapter-f16.gguf`)
        .replace('id="__x"', 'id="regAdapter"'))}
      ${Forge.field('Base GGUF', Forge.sel('regBase',
        (Forge.state.boot.base_ggufs || []).map(b => b.name)).replace('id="__x"', 'id="regBase"'))}
      <div class="row" style="justify-content:flex-end;margin-top:12px">
        <button class="btn" data-x="c">Cancel</button>
        <button class="btn primary" data-x="s">Register</button></div>`);
    box.querySelector('[data-x=c]').onclick = Forge.closeModal;
    box.querySelector('[data-x=s]').onclick = Forge.safe(async () => {
      await Forge.post('/api/register', {
        name,
        adapter_gguf: box.querySelector('#regAdapter').value,
        base_gguf: 'gguf_output/' + box.querySelector('#regBase').value,
      });
      Forge.closeModal();
      await Forge.refresh();
      Forge.ok(`${name} registered — restart the engine to load its adapter`);
    });
  }

  // ── live status ────────────────────────────────────────
  function onStatus(tr) {
    if (!tr) return;
    lastStatus = tr;
    const st = document.getElementById('trStatus');
    if (!st) return;
    st.textContent = tr.running ? (tr.status || 'training') : (tr.status || 'idle');
    st.className = 'pill ' + (tr.running ? 'accent' : tr.status === 'done' ? 'ok'
      : tr.status === 'error' ? 'crit' : '');

    const eta = tr.sps && tr.total_steps && tr.step
      ? ((tr.total_steps - tr.step) * (tr.samples / Math.max(tr.step, 1)) / tr.sps) : null;
    document.getElementById('trStats').innerHTML = `
      <div class="stat"><div class="k">Step</div><div class="v">${tr.step || 0}</div>
        <div class="n">of ${tr.total_steps || '?'}</div></div>
      <div class="stat"><div class="k">Train loss</div><div class="v">${
        tr.loss != null ? tr.loss : '—'}</div></div>
      <div class="stat"><div class="k">Eval loss</div><div class="v" style="color:var(--warn)">${
        tr.eval_loss != null ? tr.eval_loss : '—'}</div>
        <div class="n">${tr.eval_loss != null ? 'held-out' : 'no eval split'}</div></div>
      <div class="stat"><div class="k">Speed</div><div class="v sm">${
        tr.sps ? tr.sps + ' smp/s' : '—'}</div>
        <div class="n">${eta ? 'eta ' + Forge.dur(eta) : `epoch ${tr.epoch || 0}`}</div></div>`;

    document.getElementById('trBar').style.width = (tr.progress || 0) + '%';

    const log = document.getElementById('trLog');
    if (log && tr.log) {
      const at = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
      log.innerHTML = tr.log.slice(-160).map(l => {
        const c = /✓|🎉/.test(l) ? 'l-ok' : /⚠/.test(l) ? 'l-warn' : /✗|error|Error/.test(l) ? 'l-err' : '';
        return c ? `<span class="${c}">${Forge.esc(l)}</span>` : Forge.esc(l);
      }).join('\n');
      if (at) log.scrollTop = log.scrollHeight;
    }

    history = tr.loss_history || [];
    drawChart();
  }

  // ── loss chart ─────────────────────────────────────────
  function drawChart() {
    const cv = document.getElementById('lossChart');
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr;
    const c = cv.getContext('2d');
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);

    const css = getComputedStyle(document.documentElement);
    const line = css.getPropertyValue('--line').trim();
    const dim = css.getPropertyValue('--faint').trim();
    const accent = css.getPropertyValue('--accent').trim();
    const warn = css.getPropertyValue('--warn').trim();

    const pts = history.filter(p => p.loss != null);
    const evals = history.filter(p => p.eval_loss != null);
    const note = document.getElementById('lossNote');

    if (pts.length < 2) {
      c.fillStyle = dim; c.font = '12px system-ui'; c.textAlign = 'center';
      c.fillText('No loss data yet — start a run.', w / 2, h / 2);
      if (note) note.textContent = '';
      return;
    }

    const pad = { l: 46, r: 12, t: 12, b: 26 };
    const all = pts.map(p => p.loss).concat(evals.map(p => p.eval_loss));
    let lo = Math.min(...all), hi = Math.max(...all);
    const span = (hi - lo) || 1;
    lo -= span * 0.1; hi += span * 0.1;
    const maxStep = Math.max(...history.map(p => p.step)) || 1;
    const X = s => pad.l + (s / maxStep) * (w - pad.l - pad.r);
    const Y = v => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b);

    // grid + axis labels
    c.strokeStyle = line; c.lineWidth = 1;
    c.fillStyle = dim; c.font = '10px "JetBrains Mono", monospace';
    c.textAlign = 'right'; c.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
      const v = lo + (hi - lo) * (i / 4);
      const y = Math.round(Y(v)) + 0.5;
      c.beginPath(); c.moveTo(pad.l, y); c.lineTo(w - pad.r, y); c.stroke();
      c.fillText(v.toFixed(2), pad.l - 7, y);
    }
    c.textAlign = 'center'; c.textBaseline = 'top';
    for (let i = 0; i <= 4; i++) {
      const s = Math.round(maxStep * i / 4);
      c.fillText(String(s), X(s), h - pad.b + 7);
    }

    const plot = (data, key, color, width) => {
      c.strokeStyle = color; c.lineWidth = width; c.lineJoin = 'round';
      c.beginPath();
      data.forEach((p, i) => (i ? c.lineTo(X(p.step), Y(p[key])) : c.moveTo(X(p.step), Y(p[key]))));
      c.stroke();
      const last = data[data.length - 1];
      c.fillStyle = color;
      c.beginPath(); c.arc(X(last.step), Y(last[key]), 3, 0, Math.PI * 2); c.fill();
    };
    plot(pts, 'loss', accent, 1.8);
    if (evals.length >= 2) plot(evals, 'eval_loss', warn, 1.8);

    if (note) {
      if (evals.length >= 3) {
        // Eval turning up while train keeps falling is the overfitting signal
        // this project had no way to see before.
        const half = Math.floor(evals.length / 2);
        const early = evals.slice(0, half).reduce((a, p) => a + p.eval_loss, 0) / half;
        const late = evals.slice(half).reduce((a, p) => a + p.eval_loss, 0) / (evals.length - half);
        note.textContent = late > early * 1.02
          ? 'eval loss is rising — likely overfitting'
          : 'eval loss still improving';
        note.style.color = late > early * 1.02 ? warn : '';
      } else {
        note.textContent = evals.length ? '' : 'no eval split — set one above to see overfitting';
      }
    }
  }

  function pollExport() {
    // Without this, a finished export never shows up anywhere -- the file
    // exists on disk, but the "Register as character" dropdown reads from
    // Forge.state.boot, which nothing here ever re-fetched.
    clearInterval(exportTimer);
    exportTimer = setInterval(async () => {
      const s = await Forge.get('/api/gguf/status').catch(() => null);
      if (!s) return;
      if (!s.running) {
        clearInterval(exportTimer);
        if (s.status === 'error') {
          Forge.err(s.error || 'Export failed');
        } else {
          await Forge.refresh();
          Forge.ok('Export finished — ready to register');
        }
      }
    }, 2000);
  }

  async function loadRuns() {
    const name = Forge.val('trName');
    const box = document.getElementById('runsBox');
    if (!name || !box) return;
    const runs = await Forge.get(`/api/train/runs/${name}`).catch(() => []);
    box.innerHTML = runs.length
      ? `<div class="tw"><table><thead><tr><th>When</th><th class="num">Steps</th>
          <th class="num">Epochs</th><th class="num">Final loss</th><th class="num">Final eval</th></tr></thead><tbody>
          ${runs.slice().reverse().map(r => `<tr>
            <td>${Forge.ago(r.ts)}</td><td class="num">${r.steps || '—'}</td>
            <td class="num">${r.epochs || '—'}</td><td class="num">${r.final_loss ?? '—'}</td>
            <td class="num">${r.final_eval_loss ?? '—'}</td></tr>`).join('')}
        </tbody></table></div>`
      : `<p class="hint">No completed runs recorded for "${Forge.esc(name)}" yet.</p>`;
  }

  return {
    render,
    enter() { if (Forge.state.train) onStatus(Forge.state.train); },
    onStatus,
    onTheme: drawChart,
    leave() { clearInterval(exportTimer); },
  };
})());
