"""
train_worker.py — runs ONE QLoRA training run in its own process, then exits.

This is the heart of the VRAM fix. Because training happens in a child process,
the OS reclaims every byte of GPU memory the instant this process exits —
something no amount of in-process gc/empty_cache can guarantee with bitsandbytes
paged optimizers + accelerate singletons on Windows.

Invoked by pipeline.run_training():
    python train_worker.py <config.json> <status.json> <stop.flag>

While training runs:
  • a writer thread mirrors pipeline.training_state -> status.json (atomic) so the
    parent's /api/status keeps reporting live progress;
  • a watcher thread turns the parent's stop.flag into pipeline.stop_event, so the
    existing graceful stop (which still saves the adapter) works across processes;
  • everything print()ed goes to stdout, which the parent streams into its log.
"""

import os, sys, json, time, threading

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Keep heavy logs quiet and match the parent's allocator hint BEFORE torch loads.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["LORA_FORGE_TRAIN_WORKER"] = "1"

# This worker lives in workers/, but the pipeline it drives lives in core/.
# Put both core/ and the project root on the path so `import pipeline` (and
# pipeline's own bare `import store`) resolve no matter the launch cwd.
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # src/
for _p in (os.path.join(_SRC, "core"), _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    if len(sys.argv) < 4:
        print("usage: train_worker.py <config.json> <status.json> <stop.flag>")
        sys.exit(2)

    cfg_path, status_path, stop_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # Import the pipeline in THIS process so its training_state/stop_event are ours.
    import pipeline

    # Point the pipeline's status writer at the path the parent is watching.
    pipeline.IPC_STATUS = status_path

    config = _load_config(cfg_path)

    # Fresh state for this run (clears stop_event, resets counters/log).
    pipeline.reset_state(config.get("lora_name", "lora"))

    stop_evt = threading.Event()  # local "we're done" signal for the helper threads

    # ── writer: publish progress to the parent ──────────────────
    def _writer():
        while not stop_evt.is_set():
            pipeline.write_worker_status()
            time.sleep(0.3)
        pipeline.write_worker_status()  # final flush

    # ── watcher: parent's stop.flag -> pipeline.stop_event ──────
    def _watcher():
        while not stop_evt.is_set():
            if os.path.exists(stop_path):
                pipeline.stop_event.set()
                return
            time.sleep(0.25)

    tw = threading.Thread(target=_writer, daemon=True)
    tx = threading.Thread(target=_watcher, daemon=True)
    tw.start()
    tx.start()

    rc = 0
    try:
        pipeline._train_inproc(config)
    except Exception as e:
        import traceback
        pipeline.training_state.update({"status": "error", "error": str(e), "running": False})
        print("✗ worker fatal: " + str(e))
        print(traceback.format_exc())
        rc = 1
    finally:
        pipeline.training_state["running"] = False
        stop_evt.set()
        # give the writer one last tick to flush the terminal status
        time.sleep(0.4)
        pipeline.write_worker_status()

    # Exiting here is what actually frees the GPU. Use os._exit to skip any
    # atexit handlers that might re-touch CUDA and stall on Windows.
    sys.stdout.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
