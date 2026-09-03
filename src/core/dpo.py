"""
dpo_train.py — DPO refinement pass on top of an already-trained LoRA adapter,
using preference pairs collected by rank_responses.py.

Continues loras/<name>/ (must exist — SFT train it first if it was deleted;
persona_forge.py's GGUF export used to auto-delete this folder, but no longer
does) rather than starting a fresh adapter, so the character keeps everything
learned in the original SFT pass and just gets nudged toward the responses
you picked over the ones you rejected.

Run (always from the project root, not from inside tools/):
    python tools/dpo_train.py bob
"""

import os, sys, json, argparse

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# This script lives in tools/ but needs both its sibling tools (pipeline) and
# the root-level app modules (store) importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # core/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # project root


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("persona_name")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    args = ap.parse_args()

    pairs_path = os.path.join("dpo_data", f"{args.persona_name}_dpo_pairs.jsonl")
    if not os.path.exists(pairs_path):
        raise SystemExit(f"No DPO pairs at {pairs_path} -- run "
                         f"'python rank_responses.py {args.persona_name} --rounds 50' first.")

    adapter_dir = os.path.join("loras", args.persona_name)
    if not os.path.exists(adapter_dir) or not os.listdir(adapter_dir):
        raise SystemExit(
            f"No safetensors adapter at {adapter_dir} -- DPO continues an existing "
            f"adapter, it can't start from nothing. Regenerate it first: "
            f"python persona_forge.py train {args.persona_name}"
        )

    # pyarrow MUST be imported before torch/transformers on this machine — see
    # pipeline.py's _do_training for the full explanation (segfaults otherwise).
    import pyarrow  # noqa: F401

    import torch

    # trl (as pinned here) unconditionally imports torch.distributed.fsdp.FSDPModule,
    # which doesn't exist in the torch version pinned here (2.5.1 predates it). We
    # never touch FSDP (single-GPU QLoRA), so a dummy stand-in satisfies the import
    # without needing to change either pinned version.
    import torch.distributed.fsdp as _fsdp
    if not hasattr(_fsdp, "FSDPModule"):
        class FSDPModule:
            pass
        _fsdp.FSDPModule = FSDPModule

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer
    from datasets import Dataset
    import transformers as _tr, datasets as _ds
    _tr.logging.set_verbosity_error()
    _ds.logging.set_verbosity_error()

    import pipeline, store

    load_from = pipeline.resolve_model_path(args.model)
    print(f"-> Loading base model from {load_from}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        load_from, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("+ Model + tokenizer ready")

    print(f"-> Attaching existing adapter from {adapter_dir} (continuing it, not starting fresh)")
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    model.enable_input_require_grads()

    # Same system prompt rank_responses.py chats with, so the prompts DPO trains
    # on match how the character is actually invoked at inference time.
    chars = [c for c in store.list_characters()
             if c.get("name", "").lower() == args.persona_name.lower() and c.get("adapter_gguf")]
    persona_text = chars[0]["persona"] if chars else ""
    sys_prompt = (f"You are {args.persona_name}. {persona_text}\n"
                 f"Speak as {args.persona_name}. No name prefix, no quotation marks.")

    rows = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            chosen, rejected = d.get("chosen"), d.get("rejected")
            if not chosen or not rejected or chosen == rejected:
                continue
            rows.append({
                "prompt": [{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": d["prompt"]}],
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
            })
    if not rows:
        raise SystemExit("No usable DPO pairs found in " + pairs_path)
    print(f"-> {len(rows)} preference pairs loaded")
    dataset = Dataset.from_list(rows)

    eff_bs = 1 * 8  # per_device_train_batch_size * gradient_accumulation_steps
    import math
    steps_per_epoch = math.ceil(len(rows) / eff_bs)
    total_steps = steps_per_epoch * int(math.ceil(args.epochs))
    print(f"-> ~{steps_per_epoch} steps/epoch x {args.epochs} epochs "
         f"= ~{total_steps} steps total (effective batch {eff_bs})")

    dpo_args = DPOConfig(
        output_dir=adapter_dir, beta=args.beta, learning_rate=args.lr,
        num_train_epochs=args.epochs, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, max_length=1024,
        optim="paged_adamw_8bit", bf16=True, fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="no", report_to="none", seed=42,
    )

    import time
    from transformers import TrainerCallback

    class _ProgressCB(TrainerCallback):
        def on_train_begin(self, args_, state, control, **kwargs):
            self.t0 = time.time()
            print("-> Trainer entered training loop (step 0)", flush=True)
            return control

        def on_log(self, args_, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return control
            step, total = state.global_step, state.max_steps or total_steps
            elapsed = time.time() - self.t0
            rate = step / elapsed if elapsed > 0 else 0
            eta = (total - step) / rate if rate > 0 else float("nan")
            print(f"  step {step}/{total} - loss {logs['loss']:.4f} - "
                 f"{elapsed:.0f}s elapsed - ETA {eta:.0f}s", flush=True)
            return control

    trainer = DPOTrainer(model=model, args=dpo_args, train_dataset=dataset,
                         processing_class=tokenizer, callbacks=[_ProgressCB()])
    print("-> Starting DPO training...", flush=True)
    trainer.train()

    model.set_adapter("default")  # trainer adds a frozen "ref" copy internally; save only ours
    print(f"-> Saving refined adapter to {adapter_dir}")
    model.save_pretrained(adapter_dir, selected_adapters=["default"])
    tokenizer.save_pretrained(adapter_dir)
    print("DONE")


if __name__ == "__main__":
    main()
