/* characters.js — create, edit and delete characters; browse and prune their
   RAG memories, including the auto-memorize purge. */

Forge.tab('characters', (() => {
  let memories = [];
  const sel = () => Forge.state.char;

  function render(root) {
    const chars = Forge.state.boot?.characters || [];
    const boot = Forge.state.boot || {};
    const c = chars.find(x => x.id === sel()) || chars[0];
    if (c) Forge.state.char = c.id;

    root.innerHTML = `
      <h1 class="page">Characters</h1>
      <p class="page-sub">A character is a persona prompt, a trained adapter and a voice.
        The adapter is what makes it sound like the person; the prompt alone only
        gets you an impression.</p>

      <div class="row" style="margin-bottom:16px">
        <select id="chSel" style="max-width:220px">
          ${chars.map(x => `<option value="${x.id}"${x.id === sel() ? ' selected' : ''}>${
            Forge.esc(x.name)}</option>`).join('') || '<option value="">(none)</option>'}
        </select>
        <button class="btn sm" data-act="new">New character</button>
        <div class="spacer"></div>
        ${c ? `<button class="btn sm" data-act="chat">Chat with ${Forge.esc(c.name)} →</button>` : ''}
      </div>

      ${!c ? `<div class="empty"><div class="big">☺</div>
        <p><strong>No characters yet.</strong></p>
        <p>Create one to chat with the base model through a persona prompt, or
           build a real one in the Persona → Data → Train flow.</p></div>` : `

      <div class="grid c2">
        <div class="card">
          <div class="card-head"><h3>Identity</h3><div class="spacer"></div>
            <span class="pill mono">${Forge.esc(c.id)}</span></div>
          ${Forge.field('Name', `<input type="text" id="cName" value="${Forge.esc(c.name)}">`)}
          ${Forge.field('Persona', `<textarea id="cPersona" rows="10">${Forge.esc(c.persona || '')}</textarea>`,
            'the system prompt when no chat override is set')}
        </div>

        <div class="card">
          <div class="card-head"><h3>Model &amp; voice</h3></div>
          ${Forge.field('Base GGUF', Forge.sel('cBase',
            [['', '(engine default)']].concat((boot.base_ggufs || []).map(b => ['gguf_output/' + b.name, b.name])),
            c.base_gguf).replace('id="__x"', 'id="cBase"'))}
          ${Forge.field('Adapter GGUF', Forge.sel('cAdapter',
            [['', '(none — base model only)']].concat(
              (boot.adapters || []).map(a => ['gguf_output/' + a.name, a.name])),
            c.adapter_gguf).replace('id="__x"', 'id="cAdapter"'),
            c.adapter_gguf ? '' : 'without one this is just a prompt')}
          ${Forge.field('Voice', `<select id="cVoice">
            <option value="">(auto — assigned Kokoro preset)</option>
            <optgroup label="Your clones">${(boot.voices?.clones || []).map(v =>
              `<option value="clone:${v.name}"${c.voice === 'clone:' + v.name ? ' selected' : ''}>clone:${
                v.name}</option>`).join('')}</optgroup>
            <optgroup label="Built-in">${(boot.voices?.kokoro || []).map(v =>
              `<option value="${v}"${c.voice === v ? ' selected' : ''}>${v}</option>`).join('')}</optgroup>
          </select>`)}
          <div class="row" style="margin-top:6px">
            <button class="btn primary" data-act="save">Save</button>
            <button class="btn sm" data-act="voice">Voice tab →</button>
            <div class="spacer"></div>
            <button class="btn sm danger" data-act="del">Delete character</button>
          </div>
          <p class="hint" style="margin-top:10px">Changing the adapter needs an engine
            restart to take effect — the engine loads every adapter at launch.</p>
        </div>
      </div>

      <h2 class="sec">Memories</h2>
      <p class="hint" style="margin:-4px 0 12px">Retrieved by relevance before each feed post
        and injected into chat when "Inject character memories" is on.</p>
      <div class="card">
        <div class="row" style="margin-bottom:12px">
          <input type="text" id="memNew" placeholder="Add a memory…" style="flex:1">
          <button class="btn sm" data-act="addMem">Add</button>
          <div class="spacer"></div>
          <button class="btn sm danger" data-act="purge">Purge auto-saved post memories</button>
        </div>
        <div id="memList"></div>
      </div>`}`;

    Forge.acts(root, {
      new: () => newChar(),
      chat: () => Forge.go('chat'),
      voice: () => Forge.go('voice'),
      save: () => save(),
      del: () => del(),
      addMem: () => addMem(),
      delMem: (el) => delMem(el.dataset.id),
      toggleMem: (el) => toggleMem(el.dataset.id, el.checked),
      purge: () => purge(),
    });

    const s = document.getElementById('chSel');
    if (s) s.onchange = Forge.safe(async (e) => {
      Forge.state.char = e.target.value;
      render(root);
      await loadMemories();
    });
    document.getElementById('memNew')?.addEventListener('keydown', e => {
      if (e.key === 'Enter') addMem();
    });
  }

  async function loadMemories() {
    if (!sel()) return;
    memories = await Forge.get(`/api/characters/${sel()}/memories`).catch(() => []);
    const box = document.getElementById('memList');
    if (!box) return;
    if (!memories.length) {
      box.innerHTML = '<p class="hint">No memories stored.</p>';
      return;
    }
    const auto = memories.filter(m => m.text.startsWith('I once said:')).length;
    box.innerHTML = `
      ${auto ? `<p class="hint" style="color:var(--warn);margin-bottom:8px">
        ${auto} of these were auto-saved from the character's own feed posts. Feeding a
        bot its own output back as memory flattens it over a long session — purging
        them is usually the right call.</p>` : ''}
      <div class="tw"><table><tbody>
        ${memories.map(m => `<tr>
          <td style="width:26px"><input type="checkbox" data-change="toggleMem" data-id="${m.id}"${
            m.enabled ? ' checked' : ''}></td>
          <td>${Forge.esc(m.text)}</td>
          <td style="width:40px"><button class="btn sm danger" data-act="delMem" data-id="${m.id}">✕</button></td>
        </tr>`).join('')}
      </tbody></table></div>`;
  }

  async function save() {
    await Forge.put(`/api/characters/${sel()}`, {
      name: Forge.val('cName'),
      persona: Forge.val('cPersona'),
      base_gguf: Forge.val('cBase'),
      adapter_gguf: Forge.val('cAdapter'),
      voice: Forge.val('cVoice'),
    });
    await Forge.refresh();
    Forge.ok('Saved');
    render(document.getElementById('view-characters'));
    await loadMemories();
  }

  async function del() {
    const c = (Forge.state.boot.characters || []).find(x => x.id === sel());
    if (!await Forge.confirm(
      `Deletes "${c?.name}" and every memory and chat it has. The trained adapter files stay on disk.`,
      { title: 'Delete character?', okLabel: 'Delete' })) return;
    await Forge.del(`/api/characters/${sel()}`);
    Forge.state.char = null;
    await Forge.refresh();
    Forge.ok('Deleted');
    render(document.getElementById('view-characters'));
  }

  async function addMem() {
    const text = Forge.val('memNew').trim();
    if (!text) return;
    await Forge.post(`/api/characters/${sel()}/memories`, { text });
    document.getElementById('memNew').value = '';
    await loadMemories();
  }

  async function delMem(id) {
    await Forge.del(`/api/characters/${sel()}/memories/${id}`);
    await loadMemories();
  }

  async function toggleMem(id, enabled) {
    await Forge.api(`/api/characters/${sel()}/memories/${id}`,
      { method: 'PATCH', body: { enabled } });
  }

  async function purge() {
    if (!await Forge.confirm(
      'Removes every "I once said: …" memory the feed auto-saved. Memories you added by hand are kept.',
      { title: 'Purge auto memories?', okLabel: 'Purge' })) return;
    const r = await Forge.post(`/api/characters/${sel()}/memories/purge_auto`);
    Forge.ok(`Removed ${r.removed} auto-saved memories`);
    await loadMemories();
  }

  function newChar() {
    const box = Forge.modal(`
      <h3>New character</h3>
      ${Forge.field('Name', `<input type="text" id="ncName" placeholder="bob">`)}
      ${Forge.field('Persona', `<textarea id="ncPersona" rows="5"
        placeholder="You are a blunt, fast-typing chatbot who drops apostrophes and types in lowercase…"></textarea>`)}
      <p class="hint">You can attach a trained adapter after creating it.</p>
      <div class="row" style="justify-content:flex-end;margin-top:12px">
        <button class="btn" data-x="c">Cancel</button>
        <button class="btn primary" data-x="s">Create</button></div>`);
    box.querySelector('[data-x=c]').onclick = Forge.closeModal;
    box.querySelector('[data-x=s]').onclick = Forge.safe(async () => {
      const name = box.querySelector('#ncName').value.trim();
      if (!name) return Forge.err('A name is required.');
      const c = await Forge.post('/api/characters',
        { name, persona: box.querySelector('#ncPersona').value });
      Forge.state.char = c.id;
      Forge.closeModal();
      await Forge.refresh();
      render(document.getElementById('view-characters'));
      await loadMemories();
      Forge.ok(`${name} created`);
    });
  }

  return { render, async enter() { await loadMemories(); } };
})());
