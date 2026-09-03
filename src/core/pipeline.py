"""
pipeline.py — LoRA Forge engine (no Flask).

The whole "build a LoRA" pipeline: dataset normalisation, robust JSON salvage,
QLoRA training, GGUF conversion (base + adapter), LM-Studio sample generation,
and HuggingFace model download. All long-running work writes progress into the
module-level *_state dicts, which the Flask layer (app.py) polls and the
cognition layer (agent/mind) reads and drives.

Imported by app.py (routes) and cognition.py (the autonomous agent).
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json, threading, traceback, time, urllib.request, logging, re, ast, subprocess, sys, shutil
from pathlib import Path

os.environ["HF_HOME"] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "runtime", ".hf_temp_cache")

logging.basicConfig(level=logging.ERROR)
for _lib in ["transformers","trl","accelerate","datasets","peft","torch",
             "huggingface_hub","bitsandbytes","urllib3","filelock","werkzeug"]:
    logging.getLogger(_lib).setLevel(logging.ERROR)
os.environ["TQDM_DISABLE"]           = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# ─────────────────────────────────────────────────────────────
#  SHARED TRAINING STATE
# ─────────────────────────────────────────────────────────────
training_state = {
    "running": False, "status": "idle", "log": [],
    "progress": 0, "step": 0, "total_steps": 0,
    "loss": None, "eval_loss": None, "error": None, "lora_name": None,
    "epoch": 0, "samples": 0, "sps": 0, "loss_history": [],
}

dataset_state = {"examples": []}

stop_event = threading.Event()
# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

LOG_CAP = 2000  # keep memory bounded on long runs (logging_steps=1 over thousands of steps)

def log(msg):
    print(msg)
    training_state["log"].append(msg)
    if len(training_state["log"]) > LOG_CAP:
        del training_state["log"][:-LOG_CAP]

def reset_state(lora_name):
    stop_event.clear()
    training_state.update({
        "running": True, "status": "downloading", "log": [],
        "progress": 0, "step": 0, "total_steps": 0,
        "loss": None, "eval_loss": None, "error": None, "lora_name": lora_name,
        "epoch": 0, "samples": 0, "sps": 0, "loss_history": [],
    })

def resolve_model_path(model_name):
    if os.path.isdir(model_name):
        return model_name
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    auto_local_dir = os.path.join(_cfg.MODELS_DIR, safe_name)
    if os.path.isdir(auto_local_dir) and any(Path(auto_local_dir).iterdir()):
        return auto_local_dir
    if "/" in model_name:
        repo_only = model_name.split("/")[-1]
        repo_local_dir = os.path.join(_cfg.MODELS_DIR, repo_only)
        if os.path.isdir(repo_local_dir) and any(Path(repo_local_dir).iterdir()):
            return repo_local_dir
    return model_name

def _cleanup_failed_lora(lora_name):
    """Delete a partial LoRA directory left by a stopped or crashed training run."""
    if not lora_name:
        return
    lora_dir = os.path.join(_cfg.LORAS_DIR, lora_name)
    if os.path.isdir(lora_dir):
        try:
            shutil.rmtree(lora_dir, ignore_errors=True)
            log(f"🗑 Removed incomplete LoRA: {lora_dir}")
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────
#  DATA FORMAT LAYER
# ─────────────────────────────────────────────────────────────
def kill_worker_now():
    """Hard-kill the training subprocess immediately (no graceful save).
    Called by /api/stop and /api/kill_all when instant termination is needed."""
    global _worker_proc
    p = _worker_proc
    if p is None:
        return
    try:
        p.terminate()
    except Exception:
        pass
    for _ in range(30):
        if p.poll() is not None:
            break
        time.sleep(0.1)
    try:
        p.kill()
    except Exception:
        pass
        
def normalize_record(raw):
    if not isinstance(raw, dict):
        return None
    if isinstance(raw.get("messages"), list) and raw["messages"]:
        msgs = []
        for m in raw["messages"]:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role and content is not None:
                msgs.append({"role": str(role), "content": str(content)})
        if any(m["role"] == "assistant" for m in msgs):
            return {"messages": msgs}
        return None
    inst = str(raw.get("instruction", "")).strip()
    out  = str(raw.get("output", "")).strip()
    if not inst or not out:
        return None
    inp  = str(raw.get("input", "")).strip()
    # dataset.py stores reasoning as its own top-level key (that's what lets
    # the Data tab show/edit it separately from the visible answer) but this
    # is the only place alpaca-shaped rows get turned into what actually
    # trains — build_training_text only recognizes reasoning that's already
    # inside the assistant turn as a leading <think>...</think>. Without
    # this merge, any row with a separate "reasoning" key (every alpaca-
    # style row the Data tab itself produces, plus any external import
    # using that same documented key) silently trains with zero reasoning
    # and no error — the exact thing the "no reasoning found" warning below
    # is supposed to catch, defeated at the source.
    reasoning = str(raw.get("reasoning", "") or raw.get("thinking", "")).strip()
    if reasoning and not out.lstrip("\n").startswith("<think>"):
        out = f"<think>\n{reasoning}\n</think>\n\n{out}"
    user = (inst + ("\n\n" + inp if inp else "")).strip()
    msgs = [{"role": "user", "content": user},
            {"role": "assistant", "content": out}]
    sys_ = str(raw.get("system", "")).strip()
    if sys_:
        msgs.insert(0, {"role": "system", "content": sys_})
    return {"messages": msgs}


def record_preview(rec):
    if isinstance(rec, dict) and isinstance(rec.get("messages"), list):
        msgs = rec["messages"]
        user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
        asst = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "assistant"), "")
        return {"instruction": user, "input": "", "output": asst}
    if isinstance(rec, dict):
        return {"instruction": rec.get("instruction", ""),
                "input": rec.get("input", ""),
                "output": rec.get("output", "")}
    return {"instruction": "", "input": "", "output": ""}


def to_messages(rec):
    rec = rec if (isinstance(rec, dict) and rec.get("messages")) else normalize_record(rec)
    return rec["messages"] if rec else []


def build_training_text(rec, tokenizer):
    """
    Render one example to a training string, preserving <think>…</think> reasoning.
    Qwen3's chat template strips assistant reasoning, so we render the context via
    the template then append the assistant content verbatim.

    Returns (full_text, prompt_text) so the caller can mask the prompt portion out
    of the loss (train only on the assistant completion). Returns None when the
    example has no trainable assistant turn at the end.
    """
    msgs = to_messages(rec)
    if not msgs:
        return None

    if msgs[-1].get("role") != "assistant":
        # No assistant turn to learn from — nothing to compute completion loss on.
        return None

    final_content = (msgs[-1].get("content") or "").strip()
    context  = msgs[:-1]

    # Check that <think> OPENS the assistant content — not merely appears somewhere
    # inside it. Forge samples output a JSON array whose nested string values contain
    # <think> tags; a substring check would false-positive and cause apply_chat_template
    # to append a dangling <think> prefix, corrupting the training text.
    has_think = (
        final_content.lstrip("\n").startswith("<think>")
        and "</think>" in final_content
    )

    try:
        prompt = tokenizer.apply_chat_template(
            context, tokenize=False, add_generation_prompt=True,
            enable_thinking=has_think,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            context, tokenize=False, add_generation_prompt=True,
        )

    if prompt.rstrip().endswith("<think>") and final_content.startswith("<think>"):
        final_content = final_content[len("<think>"):].lstrip("\n")

    eos = tokenizer.eos_token or ""
    return prompt + final_content + eos, prompt


def tokenize_with_completion_mask(full_text, prompt_text, tokenizer, max_len):
    """
    Tokenize one example and build a labels mask that ignores the prompt tokens
    (set to -100) so the loss is computed only over the assistant completion.
    The chat template already emits special tokens as literal text, so we encode
    with add_special_tokens=False to avoid a spurious extra BOS.
    """
    enc = tokenizer(full_text, truncation=True, max_length=max_len,
                    add_special_tokens=False)
    input_ids = enc["input_ids"]
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_len,
                           add_special_tokens=False)["input_ids"]
    prompt_len = min(len(prompt_ids), len(input_ids))

    labels = list(input_ids)
    for i in range(prompt_len):
        labels[i] = -100

    # If truncation cut off the entire completion, there is nothing to learn.
    if all(l == -100 for l in labels):
        return None
    return {"input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "labels": labels}


# ─────────────────────────────────────────────────────────────
#  ROBUST JSON SALVAGE
# ─────────────────────────────────────────────────────────────

def _try_loads(s):
    try:
        return json.loads(s)
    except Exception:
        return None

def _repair_json(s):
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"//[^\n]*", "", s)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = "".join(ch for ch in s if ch >= " " or ch in "\n\t\r")
    return s

def _extract_array(text):
    start = text.find("[")
    end   = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text

def _parse_one_object(block):
    o = _try_loads(block) or _try_loads(_repair_json(block))
    if isinstance(o, dict):
        return o
    try:
        o = ast.literal_eval(block)
        if isinstance(o, dict):
            return o
    except Exception:
        pass
    def grab(key):
        m = re.search(rf'["\']?{key}["\']?\s*:\s*"((?:[^"\\]|\\.)*)"', block, re.DOTALL)
        return m.group(1) if m else None
    inst = grab("instruction")
    out  = grab("output") or grab("answer")
    if inst and out:
        return {"instruction": inst, "input": grab("input") or "",
                "reasoning": grab("reasoning") or "", "output": out}
    return None

def _salvage_objects(text):
    results, buf, depth, in_obj = [], "", 0, False
    in_str, esc = False, False
    for ch in text:
        if in_obj:
            buf += ch
        if esc:
            esc = False; continue
        if ch == "\\":
            esc = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                buf, in_obj = "{", True
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and in_obj:
                obj = _parse_one_object(buf)
                if obj:
                    results.append(obj)
                in_obj = False
    return results

def robust_parse_examples(content):
    arr  = _extract_array(content)
    data = _try_loads(arr)
    if isinstance(data, list) and data:
        return data, False
    data = _try_loads(_repair_json(arr))
    if isinstance(data, list) and data:
        return data, True
    salvaged = _salvage_objects(content)
    return salvaged, bool(salvaged)


# ─────────────────────────────────────────────────────────────
#  TRAINING THREAD
# ─────────────────────────────────────────────────────────────

def _do_training(config):
    # All training logic lives in this function so its local frame (model,
    # trainer, tokenizer, hooks) is destroyed on return — the only reliable
    # way to free bitsandbytes 4-bit VRAM on Windows.
    #
    # pyarrow MUST be imported before torch/transformers on this machine —
    # `from transformers import Trainer` pulls in sklearn -> pandas -> pyarrow
    # transitively, and importing pyarrow AFTER torch/transformers segfaults
    # (Windows access violation) due to a native library init-order conflict.
    # Importing it first here sidesteps that entirely; confirmed via
    # faulthandler traceback + minimal repro during debugging.
    import pyarrow  # noqa: F401 — import order fix, not otherwise used directly
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                               BitsAndBytesConfig, TrainerCallback,
                               Trainer, TrainingArguments, DataCollatorForSeq2Seq)
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    import transformers, datasets as _ds
    transformers.logging.set_verbosity_error()
    _ds.logging.set_verbosity_error()

    model_name   = config["model_name"]
    lora_name    = config["lora_name"]
    dataset_path = config["dataset_path"]
    output_dir   = os.path.join(_cfg.LORAS_DIR, lora_name)
    os.makedirs(output_dir, exist_ok=True)

    log(f"▶ Config: seq={config['max_seq_length']} rank={config['lora_rank']} "
        f"alpha={config['lora_alpha']} epochs={config['num_epochs']} "
        f"batch={config['batch_size']}×{config['grad_accum']}")

    # ── Load model ──────────────────────────────────────
    training_state["status"] = "downloading"
    load_from = resolve_model_path(model_name)
    log(f"▶ Loading model from {'local folder' if os.path.isdir(load_from) else 'HuggingFace'}: {load_from}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        load_from, quantization_config=bnb,
        device_map={"": 0}, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(load_from)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    log("✓ Model + tokenizer ready")

    # ── Attach LoRA ──────────────────────────────────────
    training_state["status"] = "training"
    log(f"▶ Attaching LoRA adapter: {lora_name}")
    lora_cfg = LoraConfig(
        r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"], bias="none",
        target_modules=[m.strip() for m in str(
            config.get("target_modules") or "q_proj,k_proj,v_proj,o_proj"
        ).split(",") if m.strip()],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    # Required now that we use a plain Trainer (no TRL kbit-prep): with a frozen
    # 4-bit base + gradient checkpointing, gradients won't flow unless the input
    # embeddings are made to require grad.
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log(f"✓ LoRA attached — {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")

    # ── Load dataset ──────────────────────────────────────
    log(f"▶ Loading dataset: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    rows = []
    skipped = 0
    think_count = 0
    max_len = config["max_seq_length"]
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                log(f"  ⚠ Skipping line {i+1}: {e}")
                skipped += 1
                continue
            rec = normalize_record(raw)
            if not rec:
                skipped += 1; continue
            built = build_training_text(rec, tokenizer)
            if not built:
                skipped += 1; continue
            full_text, prompt_text = built
            if "<think>" in full_text and "</think>" in full_text and not re.search(r"<think>\s*</think>", full_text):
                think_count += 1
            row = tokenize_with_completion_mask(full_text, prompt_text, tokenizer, max_len)
            if not row:
                skipped += 1; continue
            rows.append(row)

    if not rows:
        raise ValueError("Dataset is empty or all lines failed to parse.")
    log(f"✓ Loaded {len(rows)} examples" + (f" ({skipped} skipped)" if skipped else ""))
    log(f"  ℹ {think_count}/{len(rows)} examples contain non-empty <think> reasoning")
    log("  ℹ Loss is masked to the assistant completion (prompt tokens are ignored)")
    if think_count == 0:
        log("  ⚠ No reasoning found — model won't learn to think. "
            "Add <think>…</think> to your assistant outputs to preserve thinking.")

    # ── Held-out eval split ──────────────────────────────
    # Without this there is no overfitting signal at all: train loss alone
    # keeps falling on a few hundred examples long after the adapter has
    # started memorising them. eval_split_pct=0 disables it.
    eval_pct = int(config.get("eval_split_pct", 0) or 0)
    eval_rows = []
    if 0 < eval_pct < 90 and len(rows) >= 20:
        import random as _rnd
        _rnd.Random(config.get("seed", 42)).shuffle(rows)
        n_eval = max(1, int(len(rows) * eval_pct / 100))
        eval_rows, rows = rows[:n_eval], rows[n_eval:]
        log(f"  ℹ Held out {len(eval_rows)} examples ({eval_pct}%) for evaluation")
    elif eval_pct:
        log(f"  ⚠ Dataset too small for a {eval_pct}% eval split — training on all of it")

    dataset = Dataset.from_list(rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None
    log("✓ Dataset ready")

    # ── Log first 2 examples as a sanity-check preview ──────────────
    for _pi in range(min(2, len(rows))):
        _ex = rows[_pi]
        _ids = _ex["input_ids"]
        _lab = _ex["labels"]
        _full_dec = tokenizer.decode(_ids, skip_special_tokens=False)[:300]
        _prompt_len = sum(1 for l in _lab if l == -100)
        _prompt_dec = tokenizer.decode(_ids[:_prompt_len], skip_special_tokens=False)[:200]
        log(f"  📋 Sample {_pi+1} preview ({len(_ids)} tokens, {_prompt_len} prompt-masked):")
        log(f"     FULL : {_full_dec.replace(chr(10), '↵')[:240]}")
        log(f"     PROMPT: {_prompt_dec.replace(chr(10), '↵')[:180]}")
    # ─────────────────────────────────────────────────────────────────

    # ── Train ─────────────────────────────────────────────
    log("▶ Starting training...")
    eff_bs = max(1, config["batch_size"]) * max(1, config["grad_accum"])

    from transformers import TrainerCallback as _TrCB
    class _CB(_TrCB):
        def __init__(self, eff_bs):
            super().__init__()
            self.eff_bs = eff_bs
            self.t_last = None
            self.step_last = 0
        def on_train_begin(self, args, state, control, **kwargs):
            self.t_last = time.time()
            self.step_last = 0
            log("  ▷ Trainer entered training loop (step 0)")
            return control
        def on_step_end(self, args, state, control, **kwargs):
            if stop_event.is_set():
                control.should_training_stop = True
            return control
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return control
            step  = state.global_step
            total = state.max_steps or 0
            loss  = logs.get("loss", logs.get("train_loss"))
            epoch = logs.get("epoch", state.epoch or 0)
            now   = time.time()
            dt    = now - (self.t_last or now)
            dstep = step - self.step_last
            sps   = (dstep * self.eff_bs / dt) if (dt > 0 and dstep > 0) else training_state.get("sps", 0)
            self.t_last  = now
            self.step_last = step
            training_state.update({
                "step": step, "total_steps": total,
                "loss": round(loss, 4) if loss is not None else training_state.get("loss"),
                "epoch": round(float(epoch), 2),
                "samples": step * self.eff_bs,
                "sps": round(sps, 2),
                "progress": int((step / max(total, 1)) * 90),
            })
            # Record the SERIES, not just the latest scalar -- a single
            # `loss` value can't be charted, and it is the only thing that
            # used to cross the IPC boundary to the web UI.
            eval_loss = logs.get("eval_loss")
            if loss is not None or eval_loss is not None:
                hist = training_state.setdefault("loss_history", [])
                if eval_loss is not None and hist and hist[-1]["step"] == step:
                    hist[-1]["eval_loss"] = round(float(eval_loss), 4)
                else:
                    hist.append({
                        "step": step,
                        "epoch": round(float(epoch), 3),
                        "loss": round(float(loss), 4) if loss is not None else None,
                        "eval_loss": round(float(eval_loss), 4) if eval_loss is not None else None,
                        "lr": logs.get("learning_rate"),
                    })
                if len(hist) > 2000:
                    del hist[:-2000]
            if eval_loss is not None:
                training_state["eval_loss"] = round(float(eval_loss), 4)
                log(f"  eval  step {step} · eval_loss {float(eval_loss):.4f}")
            if loss is not None:
                log(f"  step {step}/{total} · epoch {float(epoch):.2f} · "
                    f"loss {loss:.4f} · {sps:.1f} smp/s · {step*self.eff_bs} samples")
            else:
                keys = ", ".join(f"{k}={logs[k]}" for k in logs if k != "epoch") or "none"
                log(f"  step {step}/{total} · (log event w/o loss → {keys})")
            return control

    # When we have a held-out eval set, track it for real: keep whichever
    # checkpoint actually had the lowest eval_loss, not just whatever the
    # weights happen to be when the last step finishes. Without this,
    # `model.save_pretrained()` below always exports the final step even
    # when eval_loss bottomed out epochs earlier and climbed back up on
    # the training set's overfitting tail (this bit the very first
    # Captain Jesus run, and the retrain after the dataset rewrite).
    # load_best_model_at_end requires save_strategy to match eval_strategy
    # and save_steps to be a round multiple of eval_steps, so when eval is
    # on we just save on the same cadence we evaluate on.
    eval_steps = max(config["logging_steps"], 10)
    best_model_kwargs = (
        {"eval_strategy": "steps", "eval_steps": eval_steps,
         "per_device_eval_batch_size": max(1, config["batch_size"]),
         "save_strategy": "steps", "save_steps": eval_steps,
         "load_best_model_at_end": True,
         "metric_for_best_model": "eval_loss", "greater_is_better": False}
        if eval_dataset is not None else
        {"save_strategy": "steps", "save_steps": config["save_steps"]}
    )
    train_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config.get("warmup_ratio", 0.03),
        lr_scheduler_type=config.get("lr_scheduler", "cosine"),
        optim="paged_adamw_8bit", fp16=False, bf16=True,
        logging_steps=config["logging_steps"],
        logging_first_step=True,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        seed=config.get("seed", 42), report_to="none",
        **best_model_kwargs,
    )
    # Pads input_ids/attention_mask/labels together and keeps label padding at -100,
    # so masked prompt tokens stay out of the loss.
    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, processing_class=tokenizer,
                      train_dataset=dataset, eval_dataset=eval_dataset,
                      args=train_args,
                      data_collator=collator, callbacks=[_CB(eff_bs)])
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint") or None)

    stopped = stop_event.is_set()
    if stopped:
        log("⚠ Stopped by user.")
        training_state.update({"status": "idle", "running": False})
    else:
        log("✓ Training complete")
        if eval_dataset is not None and getattr(trainer.state, "best_metric", None) is not None:
            log(f"  best eval_loss {trainer.state.best_metric:.4f} "
                f"(step {trainer.state.best_global_step}) — exporting that checkpoint, "
                f"not necessarily the final step")
        log(f"▶ Saving adapter to: {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        log("✓ Adapter saved (safetensors)")
        training_state.update({"progress": 100, "status": "done", "running": False})
        log(f"\n🎉 Done! LoRA '{lora_name}' is ready in loras/{lora_name}/")

    try:
        model.base_model.disable_input_require_grads()
    except Exception:
        pass
    try:
        for module in model.modules():
            module._forward_hooks.clear()
            module._forward_pre_hooks.clear()
            module._backward_hooks.clear()
    except Exception:
        pass

    return stopped


# ─────────────────────────────────────────────────────────────
#  TRAINING SUPERVISOR  (subprocess isolation — the VRAM fix)
# ─────────────────────────────────────────────────────────────
# WHY A SUBPROCESS:
#   bitsandbytes' paged optimizer (GlobalOptimManager), the accelerate
#   AcceleratorState singleton, and the CUDA caching allocator all keep VRAM
#   reserved for the life of the *process*. del-ing the model + gc.collect() +
#   torch.cuda.empty_cache() can NEVER fully reclaim it in-process on Windows —
#   which is exactly why the engine then OOM'd on restart and the auto-train run
#   cascaded into "Engine not running". Running training in a child process means
#   the OS reclaims 100% of its VRAM the instant it exits. Guaranteed. No residue.
#
# IPC: a tiny file protocol in .train_ipc/
#   config.json  — the parent writes the run config here
#   status.json  — the worker mirrors its training_state here (atomic temp+rename)
#   stop.flag    — the parent touches this; the worker turns it into stop_event

import config as _cfg

# Absolute, under runtime/, so the parent and the child worker process agree
# on these paths no matter which directory either was launched from.
IPC_DIR     = _cfg.IPC_DIR
IPC_CONFIG  = os.path.join(IPC_DIR, "config.json")
IPC_STATUS  = os.path.join(IPC_DIR, "status.json")
IPC_STOP    = os.path.join(IPC_DIR, "stop.flag")

# Fields the parent mirrors from the worker's status file into its own
# training_state so /api/status keeps reporting live progress unchanged.
_MIRROR_KEYS = ("status", "progress", "step", "total_steps", "loss",
                "error", "lora_name", "epoch", "samples", "sps", "log",
                "loss_history", "eval_loss")


def _atomic_write_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def write_worker_status():
    """Called FROM the worker process to publish progress to the parent."""
    try:
        _atomic_write_json(IPC_STATUS, {k: training_state.get(k) for k in _MIRROR_KEYS})
    except Exception:
        pass


def _read_status_file():
    try:
        with open(IPC_STATUS, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ── module-level handle so stop_event can kill it from anywhere ──────────────
_worker_proc = None   # set by run_training, cleared on exit


def run_training(config):
    """PARENT-SIDE supervisor: launch train_worker.py, mirror its progress into
    training_state, forward stop requests, and let the OS free all VRAM on exit.

    Falls back to the legacy in-process path only if launching the worker fails
    (e.g. train_worker.py missing) so a broken install still trains, just without
    the guaranteed VRAM reclaim."""
    global _worker_proc

    # reset_state() clears stop_event too, but it only ever runs INSIDE the
    # worker subprocess (train_worker.py calls it after import) -- that's a
    # different Python process with its own separate copy of this module, so
    # clearing it there does nothing to the parent's stop_event. Without this,
    # clicking Stop once sets the parent's flag permanently: every future run
    # would see it already set on its very first monitor-loop iteration and
    # kill the freshly-launched worker before it does any work at all.
    stop_event.clear()

    os.makedirs(IPC_DIR, exist_ok=True)
    for p in (IPC_STATUS, IPC_STOP, IPC_CONFIG):
        try:
            os.remove(p)
        except OSError:
            pass

    worker = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),   # src/
        "workers", "train_worker.py")
    if not os.path.exists(worker):
        log("ℹ train_worker.py not found — running training in-process (VRAM may "
            "not fully release until restart).")
        return _train_inproc(config)

    try:
        _atomic_write_json(IPC_CONFIG, config)
    except Exception as e:
        log(f"ℹ Could not write training config ({e}); running in-process.")
        return _train_inproc(config)

    env = dict(os.environ)
    env["LORA_FORGE_TRAIN_WORKER"] = "1"

    log("▶ Launching isolated training process (VRAM is freed completely on exit)…")
    try:
        # -u: unbuffered stdout so training log lines stream to the UI in real-time
        # rather than accumulating in Python's 8 KB pipe buffer and only appearing
        # all at once when the process exits or the buffer fills.
        proc = subprocess.Popen(
            [sys.executable, "-u", "-X", "utf8", worker, IPC_CONFIG, IPC_STATUS, IPC_STOP],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
            bufsize=1)
    except Exception as e:
        log(f"✗ Could not launch training worker ({e}); running in-process instead.")
        return _train_inproc(config)

    _worker_proc = proc

    # Primary log path: stream the worker's stdout into training_state["log"].
    # Requires -u on the worker invocation (above); without it stdout is block-
    # buffered and lines only appear in 8 KB chunks or at process exit.
    def _pump_stdout():
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line:
                    if line not in training_state["log"][-3:]:
                        training_state["log"].append(line)
                        if len(training_state["log"]) > LOG_CAP:
                            del training_state["log"][:-LOG_CAP]
        except Exception:
            pass

    threading.Thread(target=_pump_stdout, daemon=True).start()

    stop_forwarded = False
    last_status = None
    while proc.poll() is None:
        if stop_event.is_set() and not stop_forwarded:
            try:
                open(IPC_STOP, "w").close()
            except Exception:
                pass
            stop_forwarded = True
            log("⚠ Stop requested — terminating training process immediately…")
            try:
                proc.terminate()
            except Exception:
                pass
            for _w in range(50):          # 50 × 0.1 s = 5 s max
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
            if proc.poll() is None:
                log("  · process still alive after 5 s — sending SIGKILL…")
                try:
                    proc.kill()
                except Exception:
                    pass

        st = _read_status_file()
        if st:
            last_status = st
            for k in _MIRROR_KEYS:
                if k == "log":
                    # Secondary log path: merge any lines from the IPC status file
                    # that the stdout pump hasn't seen yet. This is a belt-and-
                    # suspenders fallback — with -u the stdout pump is the primary
                    # channel, but if a line ever slips through (e.g. stderr-only
                    # output), this catches it.
                    for line in (st.get("log") or []):
                        if line and line not in training_state["log"]:
                            training_state["log"].append(line)
                    if len(training_state["log"]) > LOG_CAP:
                        del training_state["log"][:-LOG_CAP]
                    continue
                if k in st and st[k] is not None:
                    training_state[k] = st[k]
            training_state["running"] = True
        time.sleep(0.4)

    # ── worker has exited → its VRAM is gone, reclaimed by the OS ──
    _worker_proc = None

    st = _read_status_file() or last_status or {}
    for k in _MIRROR_KEYS:
        if k == "log":
            continue
        if k in st and st[k] is not None:
            training_state[k] = st[k]

    rc = proc.returncode
    # A negative returncode on Unix means killed by signal — that's our SIGKILL,
    # which is a clean intentional stop, not a crash.
    killed_by_us = stop_event.is_set() and (rc is not None and rc < 0 or rc == 1)
    if killed_by_us:
        training_state.update({"status": "idle", "running": False})
        log("⚠ Training process killed by stop request — VRAM freed.")
        _cleanup_failed_lora(config.get("lora_name", ""))
    elif rc != 0 and (st.get("status") not in ("done", "idle")):
        msg = st.get("error") or f"training process exited abnormally (code {rc})"
        training_state.update({"status": "error", "error": msg})
        _cleanup_failed_lora(config.get("lora_name", ""))
        log(f"✗ {msg}")
    elif not st:
        training_state.update({"status": "error",
                               "error": "training process produced no status"})
        _cleanup_failed_lora(config.get("lora_name", ""))

    training_state["running"] = False
    log("✓ Training process exited — all of its VRAM has been returned to the GPU.")
    return bool(st.get("status") == "idle" or killed_by_us)


def _train_inproc(config):
    """The actual training body. Runs inside train_worker.py (child process), or
    in-process as a last-resort fallback. Returns True if stopped by the user."""
    try:
        stopped = _do_training(config)
    except Exception as e:
        training_state.update({"status": "error", "running": False, "error": str(e)})
        log(f"\n✗ Error: {e}")
        log(traceback.format_exc())
        stopped = False
    finally:
        training_state["running"] = False

    # Best-effort in-process cleanup. In the worker this barely matters (the
    # process is about to exit and the OS reclaims everything); in the fallback
    # path it's the only cleanup we get.
    print(">>> Releasing VRAM <<<")
    try:
        import torch, gc
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try: torch.cuda.ipc_collect()
            except Exception: pass
        if torch.cuda.is_available():
            used_mb     = torch.cuda.memory_allocated() / 1024**2
            reserved_mb = torch.cuda.memory_reserved()  / 1024**2
            log(f"✓ VRAM released — {used_mb:.0f} MB allocated, {reserved_mb:.0f} MB reserved by PyTorch")
    except Exception as e:
        log(f"ℹ Cleanup error (non-fatal): {e}")
    return stopped


# ─────────────────────────────────────────────────────────────
#  GGUF CONVERSION
# ─────────────────────────────────────────────────────────────

gguf_state = {
    "running": False, "status": "idle", "log": [],
    "output_path": None, "error": None
}

def glog(msg):
    print(msg)
    gguf_state["log"].append(msg)
    if len(gguf_state["log"]) > LOG_CAP:
        del gguf_state["log"][:-LOG_CAP]

def find_llama_cpp_dir(hint=""):
    check = lambda d: os.path.exists(os.path.join(str(d), "convert_hf_to_gguf.py"))
    if hint and check(hint): return hint
    candidates = [
        _cfg.LLAMA_CPP_DIR,                       # runtime/llama.cpp -- the real one
        ".", "llama.cpp", "../llama.cpp",
        os.path.expanduser("~/llama.cpp"),
        r"C:\llama.cpp", r"C:\tools\llama.cpp",
    ]
    try:
        for entry in os.scandir(".."):
            if entry.is_dir() and "llama" in entry.name.lower():
                candidates.append(entry.path)
    except Exception:
        pass
    for c in candidates:
        if check(c): return str(c)
    return None

def find_quantize_bin(llama_dir):
    for name in ["llama-quantize.exe", "llama-quantize", "quantize.exe", "quantize"]:
        for sub in ["", os.path.join("build","bin"), os.path.join("build","Release"), os.path.join("build","Debug")]:
            p = os.path.join(llama_dir, sub, name) if sub else os.path.join(llama_dir, name)
            if os.path.exists(p): return p
    return None

def run_gguf_conversion(config):
    lora_name   = config["lora_name"]
    output_name = config["output_name"].strip().replace(" ", "_")
    q_level     = config["q_level"]
    llama_hint  = config.get("llama_cpp_path", "").strip()

    lora_path = os.path.join(_cfg.LORAS_DIR, lora_name)
    out_dir   = _cfg.GGUF_DIR
    os.makedirs(out_dir, exist_ok=True)

    gguf_state.update({"running": True, "status": "merging",
                        "log": [], "output_path": None, "error": None})
    try:
        import torch, gc
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        merged_dir = os.path.join(out_dir, f"{output_name}_merged")

        if os.path.isdir(merged_dir) and any(Path(merged_dir).iterdir()):
            glog(f"✓ Reusing cached merged model: {merged_dir}")
        else:
            glog("▶ Loading base + LoRA on CPU (uses ~2× model size in RAM)...")
            model = AutoPeftModelForCausalLM.from_pretrained(
                lora_path, device_map="cpu", torch_dtype=torch.float16,
            )
            glog("▶ Merging LoRA weights into base...")
            model = model.merge_and_unload()
            glog("▶ Saving merged model (may take a few minutes)...")
            os.makedirs(merged_dir, exist_ok=True)
            model.save_pretrained(merged_dir)
            del model; gc.collect()

            base = lora_base_model(lora_name)
            if base:
                tok = AutoTokenizer.from_pretrained(resolve_model_path(base))
                tok.save_pretrained(merged_dir)
                del tok
            glog(f"✓ Merged model saved: {os.path.abspath(merged_dir)}")

        llama_dir = find_llama_cpp_dir(llama_hint)
        if not llama_dir:
            glog("")
            glog("⚠ llama.cpp not found — needed for GGUF conversion.")
            glog("  1. Download from: https://github.com/ggerganov/llama.cpp/releases")
            glog("  2. Extract it (e.g. C:\\llama.cpp)")
            glog("  3. Run:  pip install gguf sentencepiece")
            glog("  4. Paste the folder path in the llama.cpp Path field and retry.")
            glog(f"  Merged model ready at: {os.path.abspath(merged_dir)}")
            gguf_state.update({"running": False, "status": "needs_llama_cpp",
                               "output_path": os.path.abspath(merged_dir)})
            return

        glog(f"✓ llama.cpp found: {llama_dir}")

        gguf_state["status"] = "converting"
        f16_path = os.path.join(out_dir, f"{output_name}-f16.gguf")
        glog("▶ Converting to GGUF f16...")
        result = subprocess.run(
            [sys.executable, os.path.join(llama_dir, "convert_hf_to_gguf.py"),
             merged_dir, "--outtype", "f16", "--outfile", f16_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Conversion failed:\n{result.stderr[-3000:]}")
        glog("✓ f16 GGUF created")
        final_path = os.path.abspath(f16_path)

        if q_level != "f16":
            gguf_state["status"] = "quantizing"
            q_bin = find_quantize_bin(llama_dir)
            if not q_bin:
                glog(f"⚠ llama-quantize binary not found in {llama_dir}.")
                glog(f"  Quantize manually: llama-quantize \"{f16_path}\" \"{out_dir}\\{output_name}-{q_level}.gguf\" {q_level}")
            else:
                glog(f"▶ Quantizing to {q_level}...")
                q_path = os.path.join(out_dir, f"{output_name}-{q_level}.gguf")
                result = subprocess.run([q_bin, f16_path, q_path, q_level],
                                        capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Quantization failed:\n{result.stderr[-2000:]}")
                try: os.remove(f16_path)
                except Exception: pass
                final_path = os.path.abspath(q_path)
                glog(f"✓ Quantized to {q_level}")

        glog(f"\n🎉 Done!  →  {final_path}")
        glog(f"  Drop into:  C:\\Users\\<you>\\.lmstudio\\models\\<subfolder>\\")
        glog(f"  Then hit Rescan in LM Studio.")
        gguf_state.update({"running": False, "status": "done", "output_path": final_path})

    except Exception as e:
        gguf_state.update({"running": False, "status": "error", "error": str(e)})
        glog(f"\n✗ Error: {e}")
        glog(traceback.format_exc())


# ─────────────────────────────────────────────────────────────
#  LM STUDIO SAMPLE GENERATION (logic; routes live in app.py)
# ─────────────────────────────────────────────────────────────

gen_state = {"running": False, "status": "idle", "added": 0, "target": 0,
             "batch": 0, "batches": 0, "error": None}

def _strip_fences(text):
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"): part = part[4:].strip()
            if part.startswith("["): return part
    return text

def run_generation(url, model, topic, persona, total, batch_size, temperature, timeout_s, enable_thinking):
    import math
    batches = math.ceil(total / batch_size)
    gen_state.update({"running":True,"status":"generating","added":0,"target":total,
                      "batch":0,"batches":batches,"error":None})
    thinking_str = "with-reasoning" if enable_thinking else "no-think"
    print(f"\n▶ Generation: {total} examples | {batches} batches of {batch_size} | temp={temperature} | {thinking_str}")
    persona_line = f"The assistant should act as: {persona}\n" if persona.strip() else ""
    added = 0
    for i in range(batches):
        if not gen_state["running"]: break
        this_batch = min(batch_size, total - added)
        gen_state["batch"] = i + 1
        print(f"  batch {i+1}/{batches} — requesting {this_batch}...", end="", flush=True)

        if enable_thinking:
            schema = ('[{"instruction":"a user request","input":"",'
                      '"reasoning":"the assistant\'s private step-by-step thinking",'
                      '"answer":"the assistant\'s final reply to the user"}]')
            sys_msg = ("You generate fine-tuning data for a reasoning model. For each "
                       "example, put the private chain-of-thought in 'reasoning' and the "
                       "final reply in 'answer'. Output ONLY a JSON array. Escape every "
                       "quote and newline so the JSON parses.")
            user_content = (
                f"Generate exactly {this_batch} diverse training examples.\n"
                f"Topic / context: {topic}\n{persona_line}"
                f"IMPORTANT: For the output/answer, always include reasoning at the very beginning "
                f"wrapped in [thought_process] and [/thought_process] tags. "
                f"Example: [thought_process] step by step logic [/thought_process] final answer.\n"
                f"Return ONLY a valid JSON array — no markdown, no preamble:\n{schema}"
            )
            payload_dict = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_content},
                ],
                "temperature": temperature, "max_tokens": 8192,
                "chat_template_kwargs": {"enable_thinking": True},
            }
        else:
            schema = '[{"instruction":"...","input":"","output":"[thinking] reasoning here [/thinking] answer here"}]'
            user_content = (
                f"Generate exactly {this_batch} diverse training examples.\n"
                f"Topic: {topic}\n{persona_line}\n"
                f"REQUIRED: The 'output' MUST start with [thinking] and [/thinking] tags.\n"
                f"Return ONLY the JSON array:\n{schema}"
            )
            payload_dict = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a data generator. You output JSON where the 'output' field MUST contain reasoning wrapped in [thinking] tags."},
                    {"role": "user", "content": user_content},
                ],
                "temperature": temperature, "max_tokens": 8192,
                "enable_thinking": False, "thinking": {"type": "disabled"},
                "chat_template_kwargs": {"enable_thinking": False},
            }

        try:
            req = urllib.request.Request(f"{url}/v1/chat/completions",
                                         data=json.dumps(payload_dict).encode(),
                                         headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                result = json.loads(resp.read())
            raw     = result["choices"][0]["message"]["content"]
            content = _strip_fences(raw)

            examples, repaired = robust_parse_examples(content)
            if not examples:
                print(" ✗ unparseable")
                gen_state["error"] = f"Batch {i+1}: could not parse model output (salvaged 0)"
                continue

            batch_added = 0
            for ex in examples:
                if not isinstance(ex, dict): continue
                inst = str(ex.get("instruction", "")).strip()
                if not inst: continue
                if ("reasoning" in ex) or ("answer" in ex):
                    reasoning = str(ex.get("reasoning", "")).strip()
                    answer    = str(ex.get("answer", "") or ex.get("output", "")).strip()
                    if not answer: continue
                    output = f"<think>\n{reasoning}\n</think>\n\n{answer}" if reasoning else answer
                else:
                    output = str(ex.get("output", "")).strip()
                    if not output: continue
                    output = output.replace("[thinking]", "<think>").replace("[/thinking]", "</think>")

                rec = normalize_record({"instruction": inst,
                                        "input": str(ex.get("input", "")),
                                        "output": output})
                if rec:
                    dataset_state["examples"].append(rec)
                    added += 1; batch_added += 1

            gen_state["added"] = added
            tag = " (repaired)" if repaired else ""
            print(f" ✓ {batch_added} added{tag} (total: {added})")
            if batch_added == 0:
                gen_state["error"] = f"Batch {i+1}: parsed but no usable examples"
        except Exception as e:
            print(f" ✗ {e}")
            gen_state["error"] = f"Batch {i+1} failed: {e}"
    gen_state.update({"running": False, "status": "done" if added > 0 else "error"})
# ─────────────────────────────────────────────────────────────
#  API — MODEL DOWNLOAD
# ─────────────────────────────────────────────────────────────

model_dl_state = {"running": False, "status": "idle", "log": []}

def run_model_download(model_name):
    try:
        model_dl_state.update({"running": True, "status": "downloading", "log": []})
        def mlog(msg):
            print(msg); model_dl_state["log"].append(msg)
            if len(model_dl_state["log"]) > LOG_CAP:
                del model_dl_state["log"][:-LOG_CAP]

        if os.path.isdir(model_name):
            mlog(f"✓ Input is already a local folder: {model_name}")
            model_dl_state.update({"running": False, "status": "done"})
            return

        safe_name = model_name.replace("/", "_").replace("\\", "_")
        local_dir = os.path.join(_cfg.MODELS_DIR, safe_name)

        if os.path.isdir(local_dir) and any(Path(local_dir).iterdir()):
            mlog(f"✓ Model already downloaded at {local_dir}")
            model_dl_state.update({"running": False, "status": "done"})
            return

        from huggingface_hub import snapshot_download, list_repo_files

        # Only skip legacy .bin shards when the repo ALSO ships safetensors —
        # otherwise excluding *.bin would leave a .bin-only model with no weights.
        ignore = ["*.msgpack", "*.h5", "*.ot"]
        try:
            files = list_repo_files(model_name)
            if any(f.endswith(".safetensors") for f in files):
                ignore.append("*.bin")
                mlog("ℹ safetensors found — skipping redundant .bin shards")
            else:
                mlog("ℹ no safetensors in repo — downloading .bin weights")
        except Exception as e:
            mlog(f"ℹ Could not list repo files ({e}); downloading all weight formats")

        mlog(f"▶ Downloading to: {local_dir}...")
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=model_name, local_dir=local_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=ignore,
        )
        mlog(f"✓ Download complete.")
        mlog(f"🎉 Model ready at: {local_dir}")
        model_dl_state.update({"running": False, "status": "done"})
    except Exception as e:
        model_dl_state.update({"running": False, "status": "error"})
        model_dl_state["log"].append(f"✗ {e}")


# ─────────────────────────────────────────────────────────────
#  LORA / GGUF HELPERS + HF→GGUF, LoRA→adapter CONVERSIONS
# ─────────────────────────────────────────────────────────────
def lora_base_model(lora_name):
    cfg = os.path.join(_cfg.LORAS_DIR, lora_name, "adapter_config.json")
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            return json.load(f).get("base_model_name_or_path") or ""
    except Exception:
        return ""
# ── conversions: base HF → GGUF, and LoRA → adapter GGUF ────
prep_state = {"running": False, "status": "idle", "log": [], "output_path": None, "error": None}

def _plog(m):
    print(m); prep_state["log"].append(m)
    if len(prep_state["log"]) > 300:
        del prep_state["log"][:-300]

def _run_prepare_base(model_name, out_name, llama_hint, q_level="f16"):
    """Convert an HF base model to a GGUF for the hot-swap engine.

    Always converts to f16 first, then (if q_level != "f16") quantizes with
    llama-quantize. LoRA adapters apply on top of a quantized base at runtime,
    so quantizing here shrinks the shared base without breaking hot-swap.
    """
    prep_state.update({"running": True, "status": "converting", "log": [], "output_path": None, "error": None})
    try:
        llama_dir = find_llama_cpp_dir(llama_hint)
        if not llama_dir:
            raise RuntimeError("llama.cpp not found (need convert_hf_to_gguf.py).")
        src = resolve_model_path(model_name)
        os.makedirs(_cfg.GGUF_DIR, exist_ok=True)
        f16_out = os.path.abspath(os.path.join(_cfg.GGUF_DIR, f"{out_name}-base-f16.gguf"))
        _plog(f"▶ Converting base {src} → {f16_out}")
        r = subprocess.run([sys.executable, os.path.join(llama_dir, "convert_hf_to_gguf.py"),
                            src, "--outtype", "f16", "--outfile", f16_out],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-3000:])
        _plog(f"✓ Base f16 GGUF ready: {f16_out}")

        final = f16_out
        if q_level and q_level != "f16":
            prep_state["status"] = "quantizing"
            q_bin = find_quantize_bin(llama_dir)
            if not q_bin:
                _plog(f"⚠ llama-quantize binary not found in {llama_dir}; keeping f16.")
                _plog(f"  Quantize manually: llama-quantize \"{f16_out}\" "
                      f"\"gguf_output/{out_name}-base-{q_level}.gguf\" {q_level}")
            else:
                q_out = os.path.abspath(os.path.join(_cfg.GGUF_DIR, f"{out_name}-base-{q_level}.gguf"))
                _plog(f"▶ Quantizing base to {q_level} → {q_out}")
                rq = subprocess.run([q_bin, f16_out, q_out, q_level],
                                    capture_output=True, text=True)
                if rq.returncode != 0:
                    raise RuntimeError(rq.stderr[-2000:])
                try:
                    os.remove(f16_out)   # drop the large intermediate
                except Exception:
                    pass
                final = q_out
                _plog(f"✓ Base quantized to {q_level}")

        _plog(f"✓ Base GGUF ready: {final}")
        prep_state.update({"running": False, "status": "done", "output_path": final})
    except Exception as e:
        prep_state.update({"running": False, "status": "error", "error": str(e)})
        _plog(f"✗ {e}")

def _run_export_adapter(lora_name, out_name, llama_hint, outtype="f16"):
    """Convert a trained LoRA into a standalone adapter GGUF for hot-swap.

    outtype is the adapter's own precision (f16 or q8_0). Keep it high — the
    adapter is tiny, and squeezing it stacks quantization error on top of the
    (possibly quantized) base.
    """
    outtype = outtype if outtype in ("f16", "q8_0") else "f16"
    prep_state.update({"running": True, "status": "converting", "log": [], "output_path": None, "error": None})
    try:
        llama_dir = find_llama_cpp_dir(llama_hint)
        if not llama_dir:
            raise RuntimeError("llama.cpp not found (need convert_lora_to_gguf.py).")
        conv = os.path.join(llama_dir, "convert_lora_to_gguf.py")
        if not os.path.exists(conv):
            raise RuntimeError("convert_lora_to_gguf.py not found in llama.cpp — update llama.cpp.")
        lora_path = os.path.join(_cfg.LORAS_DIR, lora_name)
        os.makedirs(_cfg.GGUF_DIR, exist_ok=True)
        out = os.path.abspath(os.path.join(_cfg.GGUF_DIR, f"{out_name}-adapter-{outtype}.gguf"))
        base = lora_base_model(lora_name)
        cmd = [sys.executable, conv, lora_path, "--outtype", outtype, "--outfile", out]
        if base:
            cmd += ["--base", resolve_model_path(base)]
        _plog(f"▶ Converting adapter {lora_path} ({outtype}) → {out}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-3000:])
        _plog(f"✓ Adapter GGUF ready: {out}")
        prep_state.update({"running": False, "status": "done", "output_path": out})
        # NOTE: used to auto-delete loras/<name>/ here to save disk space, but that
        # safetensors folder is the only thing continuation training (e.g. a DPO
        # refinement pass) can load — the GGUF can't be trained on. Keep it.
    except Exception as e:
        prep_state.update({"running": False, "status": "error", "error": str(e)})
        _plog(f"✗ {e}")