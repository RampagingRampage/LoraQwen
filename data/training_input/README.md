# training_input/

Drop source documents here, one folder per persona:

```
training_input/
  <persona_name>/
    export1.txt
    export2.json      (a raw social media export — the tool digs text out of any shape)
    notes.md
```

Accepted formats: `.txt`, `.md`, `.json`, `.jsonl`, `.csv`/`.tsv`. For JSON/JSONL,
it looks for common text-ish keys (`text`, `content`, `body`, `message`,
`caption`, `post`) but falls back to flattening the whole thing, so most
export shapes work without pre-processing.

## Running it

```bash
python persona_forge.py build   <persona_name>          # documents -> master_persona.json + persona.md
python persona_forge.py gendata <persona_name> -n 300    # -> datasets/<persona_name>.jsonl
python persona_forge.py train   <persona_name>           # QLoRA train -> GGUF export -> registers character
```

`build` is resumable — it saves `master_persona.json` after every chunk, so if
it's interrupted, running `build` again picks up where it left off (merges
into what's already there rather than starting over).

Requires the local engine running (`llama-server` on port 8088) — start it via
the app's engine-start flow, or point `persona_forge.ENGINE_URL` elsewhere.
