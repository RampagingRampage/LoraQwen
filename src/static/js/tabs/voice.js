/* voice.js — the Voice tab.
   In-browser recording (retires the Tk recorder), clip management with level
   meters, reference-clip selection, and previews of every built-in Kokoro
   voice alongside the clone. */

Forge.tab('voice', (() => {
  let prompts = [];
  let clips = [];
  let voiceName = '';
  let rec = null;              // MediaRecorder
  let chunks = [];
  let stream = null;
  let analyser = null;
  let meterRaf = null;
  let recStart = 0;
  let recTimer = null;
  let currentPrompt = null;

  function render(root) {
    const boot = Forge.state.boot || {};
    voiceName = voiceName || Forge.state.dataset || (boot.voices?.clones?.[0]?.name) || '';
    root.innerHTML = `
      <h1 class="page">Voice</h1>
      <p class="page-sub">Every character already has a voice — a Kokoro preset, assigned
        deterministically. Record ten varied clips here and it can have yours instead.
        Four or five varied references beat all ten; variety matters more than volume.</p>

      <div class="row" style="margin-bottom:16px">
        <input type="text" id="vName" value="${Forge.esc(voiceName)}" placeholder="voice name"
          style="max-width:200px">
        <button class="btn sm" data-act="load">Load</button>
        <span class="pill" id="vCount"></span>
        <div class="spacer"></div>
        <button class="btn sm" data-act="upload">Upload clips</button>
      </div>

      <div class="grid c2">
        <div class="card">
          <div class="card-head"><h3>Record</h3><div class="spacer"></div>
            <span class="pill" id="recTime" hidden>0.0s</span></div>
          <div id="promptBox"></div>
          <div class="meter" style="margin:12px 0 10px"><i id="levelBar" style="width:0%"></i></div>
          <div class="row">
            <button class="btn primary" data-act="rec" id="recBtn">● Record</button>
            <button class="btn" data-act="stopRec" id="stopRecBtn" hidden>■ Stop &amp; save</button>
            <button class="btn sm" data-act="skip">Next prompt →</button>
          </div>
          <p class="hint" style="margin-top:10px">Breathing, "um", pauses and real laughs
            are good — they are what makes a clone sound like a person instead of a
            reader. Don't edit them out.</p>
        </div>

        <div class="card">
          <div class="card-head"><h3>Test</h3></div>
          ${Forge.field('Say', `<input type="text" id="vText"
            value="alright, let's see if this actually sounds like me or not.">`)}
          ${Forge.field('Voice', `<select id="vSpeaker">
            <optgroup label="Your clones" id="cloneGroup"></optgroup>
            <optgroup label="Built-in (Kokoro)">${
              (boot.voices?.kokoro || []).map(v => `<option value="${v}">${v}</option>`).join('')}
            </optgroup></select>`)}
          <div class="row">
            <button class="btn primary" data-act="preview">Speak it</button>
            <audio id="vAudio" controls style="flex:1;height:32px"></audio>
          </div>
          <h3 class="sub">Assign to a character</h3>
          <div class="row">
            ${Forge.sel('vChar', (boot.characters || []).map(c => [c.id, c.name]), Forge.state.char)
              .replace('id="__x"', 'id="vChar" style="max-width:170px"')}
            <button class="btn sm" data-act="assign">Use this voice</button>
          </div>
        </div>
      </div>

      <h2 class="sec">Clips</h2>
      <p class="hint" style="margin:-4px 0 12px">Tick the ones XTTS should clone from.
        This writes <code>refs.json</code> — hand-edited JSON until now.</p>
      <div id="clipList"></div>`;

    Forge.acts(root, {
      load: Forge.safe(async () => { voiceName = Forge.val('vName').trim(); await loadClips(); }),
      rec: () => startRec(),
      stopRec: () => stopRec(),
      skip: () => { nextPrompt(); },
      preview: () => preview(),
      assign: () => assign(),
      upload: () => uploadDialog(),
      play: (el) => { new Audio(el.dataset.url).play(); },
      delClip: (el) => delClip(el.dataset.f),
      toggleRef: () => saveRefs(),
    });
  }

  // ── clips ──────────────────────────────────────────────
  async function loadClips() {
    if (!voiceName) return;
    const d = await Forge.get(`/api/voice/samples/${voiceName}`).catch(() => ({ clips: [] }));
    clips = d.clips || [];
    paintClips();
    paintCloneGroup();
    const c = document.getElementById('vCount');
    if (c) c.textContent = `${clips.length} clips · ${clips.filter(x => x.is_ref).length} used as references`;
  }

  function paintCloneGroup() {
    const g = document.getElementById('cloneGroup');
    if (!g) return;
    const clones = Forge.state.boot?.voices?.clones || [];
    g.innerHTML = clones.map(c => `<option value="clone:${c.name}">clone:${c.name} (${c.clips} clips)</option>`).join('')
      || '<option disabled>none recorded yet</option>';
  }

  function paintClips() {
    const box = document.getElementById('clipList');
    if (!box) return;
    if (!clips.length) {
      box.innerHTML = `<div class="empty"><div class="big">🎙</div>
        <p><strong>No clips for "${Forge.esc(voiceName || '—')}" yet.</strong></p>
        <p>Record the prompts above, or upload existing audio.</p></div>`;
      return;
    }
    box.innerHTML = clips.map(c => `
      <div class="clip${c.is_ref ? ' is-ref' : ''}">
        <input type="checkbox" data-change="toggleRef" class="refBox" value="${Forge.esc(c.file)}"${
          c.is_ref ? ' checked' : ''} title="Use as an XTTS reference">
        <div>
          <div class="row tight" style="margin-bottom:3px">
            <strong style="font-size:13px">${Forge.esc(c.file)}</strong>
            <span class="pill">${c.seconds ?? '?'}s</span>
            <span class="pill">peak ${c.peak_db ?? '?'} dB</span>
            <span class="pill">rms ${c.rms_db ?? '?'} dB</span>
            ${(c.notes || []).map(n => `<span class="pill warn">${Forge.esc(n)}</span>`).join('')}
            ${c.ok ? '<span class="pill ok">good</span>' : ''}
          </div>
          <div class="meter"><i style="width:${Math.min(100, ((c.peak_db ?? -60) + 60) / 60 * 100)}%${
            (c.peak_db ?? -60) > -1 ? ';background:var(--crit)' : ''}"></i></div>
        </div>
        <div class="row tight">
          <button class="btn sm" data-act="play" data-url="${Forge.esc(c.url)}">▶</button>
          <button class="btn sm danger" data-act="delClip" data-f="${Forge.esc(c.file)}">✕</button>
        </div>
      </div>`).join('');
  }

  async function saveRefs() {
    const chosen = Forge.$$('.refBox').filter(b => b.checked).map(b => b.value);
    await Forge.put(`/api/voice/samples/${voiceName}/refs`, { clips: chosen });
    Forge.ok(`${chosen.length} reference clips saved`);
    await loadClips();
  }

  async function delClip(f) {
    if (!await Forge.confirm(`Deletes ${f} permanently.`,
      { title: 'Delete clip?', okLabel: 'Delete' })) return;
    await Forge.del(`/api/voice/samples/${voiceName}/${f}`);
    await loadClips();
    await Forge.refresh();
  }

  // ── prompts ────────────────────────────────────────────
  function paintPrompt() {
    const box = document.getElementById('promptBox');
    if (!box || !currentPrompt) return;
    const done = clips.some(c => c.file.startsWith(String(currentPrompt.index).padStart(2, '0')));
    box.innerHTML = `
      <div class="row tight" style="margin-bottom:6px">
        <span class="pill accent">${currentPrompt.index} of ${prompts.length}</span>
        <span class="pill">${Forge.esc(currentPrompt.label)}</span>
        ${done ? '<span class="pill ok">recorded</span>' : ''}
      </div>
      <p style="font-size:14px;line-height:1.6;margin:0">${Forge.esc(currentPrompt.prompt)}</p>
      <p class="hint">Saves as <code>${Forge.esc(currentPrompt.filename)}</code></p>`;
  }

  function nextPrompt() {
    if (!prompts.length) return;
    const i = prompts.indexOf(currentPrompt);
    currentPrompt = prompts[(i + 1) % prompts.length];
    paintPrompt();
  }

  // ── recording ──────────────────────────────────────────
  async function startRec() {
    if (!voiceName) return Forge.err('Name the voice first.');
    if (rec) return;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false,
                 autoGainControl: false },
      });
    } catch (e) {
      return Forge.err('Microphone access was denied or unavailable.');
    }

    // Live level meter — so you can see you're too quiet before recording ten
    // clips you have to redo.
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    ctx.createMediaStreamSource(stream).connect(analyser);
    const buf = new Uint8Array(analyser.fftSize);
    const bar = document.getElementById('levelBar');
    const tick = () => {
      analyser.getByteTimeDomainData(buf);
      let peak = 0;
      for (const v of buf) peak = Math.max(peak, Math.abs(v - 128) / 128);
      bar.style.width = Math.min(100, peak * 140) + '%';
      bar.classList.toggle('hot', peak > 0.95);
      meterRaf = requestAnimationFrame(tick);
    };
    tick();

    chunks = [];
    rec = new MediaRecorder(stream);
    rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    rec.onstop = Forge.safe(saveRecording);
    rec.start();

    recStart = Date.now();
    document.getElementById('recBtn').hidden = true;
    document.getElementById('stopRecBtn').hidden = false;
    const t = document.getElementById('recTime');
    t.hidden = false;
    recTimer = setInterval(() => {
      t.textContent = ((Date.now() - recStart) / 1000).toFixed(1) + 's';
    }, 100);
  }

  function stopRec() {
    if (rec && rec.state !== 'inactive') rec.stop();
  }

  async function saveRecording() {
    clearInterval(recTimer);
    cancelAnimationFrame(meterRaf);
    stream?.getTracks().forEach(t => t.stop());
    document.getElementById('recBtn').hidden = false;
    document.getElementById('stopRecBtn').hidden = true;
    document.getElementById('recTime').hidden = true;
    document.getElementById('levelBar').style.width = '0%';
    const blob = new Blob(chunks, { type: rec.mimeType || 'audio/webm' });
    rec = null;

    if (blob.size < 2000) return Forge.err('That recording was empty.');
    const fd = new FormData();
    fd.append('label', currentPrompt ? currentPrompt.filename.replace('.wav', '') : 'clip');
    fd.append('file', blob, 'rec.webm');
    try {
      const r = await Forge.api(`/api/voice/samples/${voiceName}`, { method: 'POST', body: fd });
      Forge.ok(`Saved ${r.file} · ${r.seconds}s` + (r.notes?.length ? ` (${r.notes.join(', ')})` : ''));
      await loadClips();
      await Forge.refresh();
      nextPrompt();
    } catch (e) {
      Forge.err(e);
    }
  }

  function uploadDialog() {
    const box = Forge.modal(`
      <h3>Upload clips</h3>
      <p class="hint" style="margin-bottom:12px">Anything ffmpeg can read. Converted to
        44.1 kHz mono 16-bit WAV, which is what XTTS wants.</p>
      ${Forge.field('Files', `<input type="file" id="upFiles" accept="audio/*" multiple>`)}
      <div class="row" style="justify-content:flex-end">
        <button class="btn" data-x="c">Cancel</button>
        <button class="btn primary" data-x="s">Upload</button></div>`);
    box.querySelector('[data-x=c]').onclick = Forge.closeModal;
    box.querySelector('[data-x=s]').onclick = Forge.safe(async () => {
      const files = [...box.querySelector('#upFiles').files];
      if (!files.length) return;
      for (const f of files) {
        const fd = new FormData();
        fd.append('label', f.name.replace(/\.[^.]+$/, ''));
        fd.append('file', f, f.name);
        await Forge.api(`/api/voice/samples/${voiceName}`, { method: 'POST', body: fd });
      }
      Forge.closeModal();
      Forge.ok(`${files.length} clip(s) uploaded`);
      await loadClips();
      await Forge.refresh();
    });
  }

  // ── preview + assign ───────────────────────────────────
  async function preview() {
    const r = await Forge.post('/api/voice/preview', {
      text: Forge.val('vText'), speaker: Forge.val('vSpeaker'),
    });
    const a = document.getElementById('vAudio');
    a.src = r.url;
    a.play().catch(() => {});
  }

  async function assign() {
    const cid = Forge.val('vChar');
    if (!cid) return Forge.err('No character selected.');
    await Forge.put(`/api/characters/${cid}`, { voice: Forge.val('vSpeaker') });
    await Forge.refresh();
    Forge.ok('Voice assigned');
  }

  return {
    render,
    async enter() {
      prompts = await Forge.get('/api/voice/prompts');
      currentPrompt = currentPrompt || prompts[0];
      paintPrompt();
      await loadClips();
    },
    leave() {
      stopRec();
      cancelAnimationFrame(meterRaf);
      stream?.getTracks().forEach(t => t.stop());
    },
  };
})());
