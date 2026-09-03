/* help.js — tips written against the failure modes this project actually hit,
   plus a VRAM table and a glossary. */

Forge.tab('help', (() => {
  const TROUBLE = [
    ['Loss went down but it still sounds like the base model',
     `Three usual causes. Rank or alpha too low for how distinct the voice is — try
      rank 32 / alpha 64. The adapter isn't actually loaded in the engine — check the
      Engine tab's adapter list, and remember adapters are loaded at launch, so a newly
      exported one needs a restart. Or the dataset is too generic: run the Evaluate tab's
      style scorecard against it, and if the <em>dataset itself</em> doesn't match the
      persona, no amount of training will fix that.`],

    ['Generation stops early and every new sample is a duplicate',
     `A diversity wall, not a bug. The generator cycles through every (topic × message-type)
      combination in the profile before repeating, so once that space is exhausted it
      genuinely has nothing new to say. Widen the topic pool in the persona profile, or
      raise the temperature. It stops on its own after five empty batches rather than
      burning cycles producing rejects.`],

    ['The engine won\'t start after a training run',
     `VRAM. bitsandbytes' paged optimizer, accelerate's state singleton and the CUDA
      caching allocator all hold memory for the life of the process, and no amount of
      <code>del</code> + <code>empty_cache()</code> reliably gets it back in-process on
      Windows. That is exactly why training runs in a child process — the OS reclaims
      everything when it exits. If it still fails, check for an orphaned
      <code>llama-server.exe</code>.`],

    ['llama-server is still running after I closed everything',
     `A Windows quirk: the engine process outlives its parent. Use "Kill orphans" in the
      Engine tab, or <code>taskkill /F /IM llama-server.exe</code>. Always check this
      before assuming an OOM is real — two engines fighting over the GPU looks exactly
      like a memory problem.`],

    ['The cloned voice sounds robotic or wrong',
     `Reference clips too clean, too few, or too similar to each other. XTTS wants natural
      range: an excited clip, a tired one, a laugh, an explanation. Breathing and "um"s
      help. Four or five varied clips beat all ten near-identical ones. Check the level
      meters in the Voice tab — a clip flagged "quiet" or "clipping" hurts the clone
      more than a short one does.`],

    ['A crash with no output, right around a Trainer import',
     `The pyarrow import-order bug. <code>from transformers import Trainer</code> pulls
      in sklearn → pandas → pyarrow, and importing pyarrow <em>after</em> torch segfaults
      on this machine — a native init-order conflict, confirmed with faulthandler. The
      pipeline imports pyarrow first to sidestep it. If you add a new training entry
      point, do the same.`],

    ['The character forgets who it is over a long feed session',
     `Auto-memorize. It used to save every post back as a memory, which RAG then retrieved
      as context for the next post — a loop that flattens the character. It's off by
      default now. If you turned it on, purge the accumulated ones from the Characters
      tab.`],

    ['Rows silently vanish from training',
     `They were over the sequence length and got truncated, or failed to parse. The Data
      tab's analysis reports both — check "Over length" and "Unparseable" before every
      run. The training log also prints how many examples it actually loaded versus how
      many lines the file had.`],
  ];

  const VRAM = [
    ['8',  '512',  '1', '~7 GB',  'ok'],
    ['16', '1024', '2', '~9 GB',  'ok'],
    ['16', '2048', '2', '~12 GB', 'warn'],
    ['32', '1024', '2', '~10 GB', 'ok'],
    ['32', '2048', '4', '~19 GB', 'crit'],
    ['64', '4096', '4', '~30 GB', 'crit'],
  ];

  const GLOSSARY = [
    ['LoRA', 'A small set of extra weights trained on top of a frozen base model. Teaches a manner, not facts.'],
    ['Rank', 'How much capacity the adapter has. Higher learns more nuance and uses more VRAM. 16 is a good default; 8 for a simple voice, 32+ for a distinctive one.'],
    ['Alpha', 'A scaling factor on the adapter\'s contribution. Conventionally 2× the rank.'],
    ['QLoRA', 'LoRA on a 4-bit quantized base. What makes training an 8B model on one consumer GPU possible.'],
    ['Epoch', 'One full pass over the dataset. Three is typical; more memorizes.'],
    ['GGUF', 'llama.cpp\'s model file format. Both the base model and each exported adapter are GGUFs.'],
    ['Quantization', 'Storing weights at lower precision. Q8 is near-lossless and large; Q4 is small and noticeably degraded.'],
    ['SFT', 'Supervised fine-tuning — the normal training pass, learning from prompt/response pairs.'],
    ['DPO', 'Direct Preference Optimization — a second pass that learns from chosen-over-rejected pairs. Refines a decent adapter; it cannot rescue a bad one.'],
    ['RAG', 'Retrieval-augmented generation. Memories are embedded and the relevant ones are pulled into context before generating.'],
    ['Eval split', 'Rows held out of training and scored separately. The only signal that tells you the model is memorizing rather than learning.'],
    ['Perplexity', 'How surprised the model is by text it should find familiar. Lower is better; comparable only between runs on the same data.'],
  ];

  function render(root) {
    root.innerHTML = `
      <h1 class="page">Help</h1>
      <p class="page-sub">Written against the things that actually went wrong here, not
        generic advice. If something is behaving strangely, it is probably on this page.</p>

      <h2 class="sec">Troubleshooting</h2>
      ${TROUBLE.map(([q, a]) => `
        <details class="card" style="cursor:pointer">
          <summary style="font-weight:600;font-size:13.5px">${Forge.esc(q)}</summary>
          <p style="margin:10px 0 0;font-size:13px;max-width:76ch;color:var(--ink-2)">${a}</p>
        </details>`).join('')}

      <h2 class="sec">Rules of thumb</h2>
      <div class="grid c2">
        <div class="card">
          <h3 class="sub" style="margin-top:0">Dataset</h3>
          <ul style="font-size:13px;color:var(--ink-2);padding-left:18px;margin:0">
            <li>200 good examples beat 2,000 repetitive ones.</li>
            <li>Outputs should be as short as the person actually writes. Long, polished
                answers are the most common reason a voice adapter sounds generic.</li>
            <li>Dedupe threshold 0.62 is a reasonable default; drop it to 0.5 if
                everything still sounds samey.</li>
            <li>Set supervision to 100% for the first 20 samples of a new persona — you
                will catch a bad profile in two minutes instead of after a training run.</li>
          </ul>
        </div>
        <div class="card">
          <h3 class="sub" style="margin-top:0">Training</h3>
          <ul style="font-size:13px;color:var(--ink-2);padding-left:18px;margin:0">
            <li>Always set an eval split. 10% is enough to see overfitting.</li>
            <li>If eval loss turns up while train loss keeps falling, stop — the previous
                checkpoint is better than the final one.</li>
            <li>Stop the engine before training. Two processes on one GPU is the most
                common OOM.</li>
            <li>Adding <code>gate_proj,up_proj,down_proj</code> to the target modules
                roughly doubles adapter size and helps a strongly distinctive voice.</li>
          </ul>
        </div>
      </div>

      <h2 class="sec">VRAM by configuration</h2>
      <p class="hint" style="margin:-4px 0 10px">Rough figures for an 8B base in 4-bit with
        gradient checkpointing on. Measure, don't trust.</p>
      <div class="tw"><table>
        <thead><tr><th class="num">Rank</th><th class="num">Seq length</th><th class="num">Batch</th>
          <th class="num">Estimated</th><th>On a 12 GB card</th></tr></thead>
        <tbody>${VRAM.map(([r, s, b, v, k]) => `<tr>
          <td class="num">${r}</td><td class="num">${s}</td><td class="num">${b}</td>
          <td class="num">${v}</td>
          <td><span class="pill ${k}">${k === 'ok' ? 'fits' : k === 'warn' ? 'tight' : "won't fit"}</span></td>
        </tr>`).join('')}</tbody>
      </table></div>

      <h2 class="sec">Glossary</h2>
      <div class="tw"><table><tbody>
        ${GLOSSARY.map(([t, d]) => `<tr><td style="width:130px"><strong>${Forge.esc(t)}</strong></td>
          <td>${Forge.esc(d)}</td></tr>`).join('')}
      </tbody></table></div>

      <h2 class="sec">Keyboard</h2>
      <div class="tw"><table><tbody>
        <tr><td style="width:130px"><code>Enter</code></td><td>Send the message / post</td></tr>
        <tr><td><code>Shift + Enter</code></td><td>Newline instead of sending</td></tr>
        <tr><td><code>Esc</code></td><td>Close the open dialog</td></tr>
      </tbody></table></div>

      <h2 class="sec">Running it</h2>
      <div class="card">
        <p style="font-size:13px;margin:0 0 8px;color:var(--ink-2)">
          <code>lora_env\Scripts\python src\app.py</code> starts the app — see the
          README's Setup section for the full install sequence (venvs, dependencies,
          llama.cpp). The Engine tab's "Kill orphans" shuts down a stray
          <code>llama-server.exe</code> if the app didn't exit cleanly.</p>
        <p style="font-size:13px;margin:0;color:var(--ink-2)">
          Every port, path and default lives in <code>.env</code> at the project root —
          nothing is hardcoded in the source any more.</p>
      </div>`;
  }

  return { render };
})());
