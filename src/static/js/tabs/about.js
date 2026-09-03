/* about.js — how a persona actually gets made, start to finish.
   Each step links to the tab that does it and names the real file it writes. */

Forge.tab('about', (() => {
  const STEPS = [
    ['Collect', 'persona',
     `Put raw material in your own words into <code>training_input/&lt;name&gt;/</code> — chat
      exports, posts, anything unedited. Thousands of lines beats a curated few hundred:
      the distiller is looking for how you actually write, including the parts you would
      tidy up if you were writing for an audience.`,
     'training_input/&lt;name&gt;/', 'Persona tab'],

    ['Filter', 'persona',
     `Strip URLs, phone numbers and lines under about twelve characters. You get a
      before/after count, so you can see how much of the corpus actually survives before
      committing to a distillation run over it.`,
     'user_responses_filtered.txt', 'Persona tab'],

    ['Distill', 'persona',
     `The model reads your text in ~1,400-character chunks and merges what it learns into a
      structured profile — typing style, vocabulary, humour, temperament, values, topics,
      verbal tics, plus verbatim sample lines. You get a short prompt version too. Both
      are editable by hand afterwards, and worth editing: a wrong trait here shows up in
      every single generated sample.`,
     'master_persona.json + persona.md', 'Persona tab'],

    ['Generate', 'data',
     `Synthetic prompt/response pairs written in your voice, deduped as they are produced.
      The generator cycles through every (topic × message-type) combination before
      repeating any — that forced variety is the actual diversity mechanism, not the
      model's own judgment of what counts as different. <strong>This is where the
      supervision slider applies:</strong> set it above zero and that fraction is held for
      you to accept, edit, regenerate or reject before it enters the dataset.`,
     'datasets/&lt;name&gt;.jsonl', 'Data tab'],

    ['Clean', 'data',
     `Dedupe, drop the empties and the over-length rows, scrub anything personal, fix or
      regenerate individual rows. Small and good beats big and noisy — this is the step
      most likely to decide whether the result sounds like you. Every operation snapshots
      the file first, so nothing here is unrecoverable.`,
     'the same file, fewer and better rows', 'Data tab'],

    ['Train', 'train',
     `QLoRA over a 4-bit base — roughly 0.3% of the weights are trainable. Set an eval
      split and watch <em>both</em> curves: train loss falling while eval loss turns back
      up is overfitting, and at a few hundred examples that happens early. Training runs
      in a child process so all its VRAM comes back when it exits.`,
     'loras/&lt;name&gt;/', 'Train tab'],

    ['Export and register', 'engine',
     `Convert the adapter to GGUF and register it as a character. One engine process holds
      the base model plus every character's adapter at once and activates whichever one is
      speaking, so adding a character does not cost another copy of the 8 GB base.`,
     'gguf_output/&lt;name&gt;-adapter-f16.gguf', 'Engine tab'],

    ['Evaluate', 'evaluate',
     `Generate side by side against the base model and previous adapters, score the output
      against the training data's measurable style, and check whether it is reciting rows
      rather than generalising. Pin a regression prompt set and re-run it after every
      training pass.`,
     'evals/runs/', 'Evaluate tab'],

    ['Refine', 'data',
     `Rank real responses against each other — four answers to the same prompt at spread
      temperatures — then run a DPO pass on what you picked. Optional, and the largest
      quality gain per unit of effort once the first training pass is decent.`,
     'dpo_data/', 'Data → DPO ranking'],
  ];

  function render(root) {
    const b = Forge.state.boot || {};
    const bob = (b.characters || [])[0];
    root.innerHTML = `
      <h1 class="page">How a persona gets made</h1>
      <p class="page-sub">Nine steps from a folder of your own writing to a model that talks
        like you and sounds like you. Each one links to the tab that does it.</p>

      <div class="card">
        <div class="card-head"><h3>The shape of it</h3></div>
        <p style="font-size:13.5px;max-width:74ch;margin:0">
          A LoRA adapter is a small set of extra weights — a few tens of megabytes — trained
          on top of a frozen base model. It does not teach the model new facts; it teaches it
          a <em>manner</em>. That is why the pipeline spends most of its effort on the dataset
          rather than the training: the adapter will faithfully learn whatever voice is in
          those few hundred examples, including the ways they are wrong.</p>
      </div>

      <h2 class="sec">The steps</h2>
      <ol class="steps">
        ${STEPS.map(([title, tab, body, out, where]) => `
          <li>
            <span class="st">${Forge.esc(title)}</span>
            <p>${body}</p>
            <span class="out">${out}</span>
            <button class="go" data-act="go" data-t="${tab}">${Forge.esc(where)} →</button>
          </li>`).join('')}
      </ol>

      <h2 class="sec">The voice track, in parallel</h2>
      <div class="card">
        <p style="font-size:13.5px;max-width:74ch;margin:0 0 10px">
          Text and voice are independent — you can do either without the other. Every
          character already has a voice: a Kokoro preset, assigned deterministically from
          its id, so it sounds like <em>something</em> from the moment it exists. Recording
          replaces that with a clone of you.</p>
        <ol class="steps">
          <li><span class="st">Record ten varied clips</span>
            <p>The prompts pull a range of tone and pace out of you — excited, frustrated,
               explaining, laughing. Breathing, "um"s and real laughs are the point, not
               noise to edit out.</p>
            <span class="out">voice_samples/&lt;name&gt;/</span>
            <button class="go" data-act="go" data-t="voice">Voice tab →</button></li>
          <li><span class="st">Pick four or five as references</span>
            <p>XTTS does not need all ten. Variety across the chosen clips matters much
               more than how many there are.</p>
            <span class="out">refs.json</span></li>
          <li><span class="st">Set the character's voice to clone:&lt;name&gt;</span>
            <p>From then on the feed and any spoken output use your cloned voice instead
               of the preset.</p>
            <span class="out">characters/&lt;id&gt;/config.json</span></li>
        </ol>
      </div>

      ${bob ? `
      <h2 class="sec">Worked example — ${Forge.esc(bob.name)}</h2>
      <div class="grid c4">
        <div class="stat"><div class="k">Dataset</div><div class="v">${
          (b.datasets || []).find(d => d.name === bob.name)?.rows ?? '—'}</div>
          <div class="n">examples</div></div>
        <div class="stat"><div class="k">Rank / alpha</div><div class="v sm">${
          (b.loras || []).find(l => l.name === bob.name)?.rank ?? '—'} / ${
          (b.loras || []).find(l => l.name === bob.name)?.alpha ?? '—'}</div></div>
        <div class="stat"><div class="k">Voice</div><div class="v sm">${
          Forge.esc(bob.voice || 'preset')}</div></div>
        <div class="stat"><div class="k">Adapter</div><div class="v sm" style="color:var(--${
          bob.adapter_gguf ? 'ok' : 'dim'})">${bob.adapter_gguf ? 'exported' : 'none'}</div></div>
      </div>
      <p class="hint">These numbers come from your actual project, not an example.</p>` : ''}

      <h2 class="sec">Where everything lives</h2>
      <div class="card">
        <div class="tw"><table><thead><tr><th>Folder</th><th>What's in it</th></tr></thead><tbody>
          <tr><td class="mono">training_input/</td><td>Raw documents and the distilled profile, per persona</td></tr>
          <tr><td class="mono">datasets/</td><td>Training data (<code>.jsonl</code>), plus review queues and snapshots</td></tr>
          <tr><td class="mono">loras/</td><td>Trained adapters and their checkpoints</td></tr>
          <tr><td class="mono">gguf_output/</td><td>The base model GGUF and every exported adapter</td></tr>
          <tr><td class="mono">characters/</td><td>Character configs, chats, and the shared RAG store</td></tr>
          <tr><td class="mono">voice_samples/</td><td>Recorded clips and the reference selection</td></tr>
          <tr><td class="mono">dpo_data/</td><td>Preference picks and derived training pairs</td></tr>
          <tr><td class="mono">evals/</td><td>Regression prompt set and past run results</td></tr>
          <tr><td class="mono">core/ · workers/</td><td>The application itself</td></tr>
        </tbody></table></div>
      </div>`;

    Forge.acts(root, { go: (el) => Forge.go(el.dataset.t) });
  }

  return { render };
})());
