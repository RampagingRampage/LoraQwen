"""verify.py — is this install actually going to work?

Checks each dependency and path the app needs and prints one line per item,
so a broken install is one glance rather than a stack trace on first use.
"""
import importlib
import os
import sys
import warnings

# Flask 3.1 deprecates __version__ and we only read it for display.
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OK, BAD, WARN = "[OK]", "[X] ", "[!] "
failures = 0


def line(name, status, note=""):
    print(f"  {name:<28}{status} {note}")


def check_import(label, module, note_ok="", required=True):
    global failures
    try:
        m = importlib.import_module(module)
        try:
            from importlib.metadata import version as _v
            v = _v(module.replace("_", "-"))
        except Exception:
            v = getattr(m, "__version__", "")
        line(label, OK, (f"v{v} " if v else "") + note_ok)
        return m
    except Exception as e:
        if required:
            failures += 1
        line(label, BAD if required else WARN, str(e)[:60])
        return None


print()
line("Python", OK if sys.version_info[:2] == (3, 11) else WARN,
     ".".join(map(str, sys.version_info[:3])) +
     ("" if sys.version_info[:2] == (3, 11) else " (3.11 is what this is tested on)"))

check_import("flask", "flask")
torch = check_import("torch", "torch")
check_import("transformers", "transformers")
check_import("peft", "peft")
check_import("datasets", "datasets")
check_import("bitsandbytes", "bitsandbytes")
check_import("numpy", "numpy")
check_import("pyarrow", "pyarrow", "must import before torch")
check_import("sentence-transformers", "sentence_transformers", required=False)
check_import("kokoro-onnx", "kokoro_onnx", required=False)

if torch is not None:
    if torch.cuda.is_available():
        line("CUDA", OK, f"{torch.cuda.get_device_name(0)} · "
                         f"{torch.cuda.get_device_properties(0).total_memory // 2**30} GB")
    else:
        failures += 1
        line("CUDA", BAD, "not available — training will not run")

try:
    import config
    line("config / .env", OK, os.path.basename(config.PROJECT_ROOT))
    for label, path, required in (
        ("llama-server.exe", os.path.join(config.LLAMA_CPP_DIR, "llama-server.exe"), True),
        ("convert_hf_to_gguf", os.path.join(config.LLAMA_CPP_DIR, "convert_hf_to_gguf.py"), True),
        ("convert_lora_to_gguf", os.path.join(config.LLAMA_CPP_DIR, "convert_lora_to_gguf.py"), True),
        ("gguf-py", os.path.join(config.LLAMA_CPP_DIR, "gguf-py"), True),
        ("base model (HF)", os.path.join(config.MODELS_DIR, "Qwen_Qwen3-8B"), False),
        ("voice_env", os.path.join(config.VOICE_ENV, "Scripts", "python.exe"), False),
        ("xtts_worker", os.path.join(config.SRC_ROOT, "workers", "xtts_worker.py"), False),
        ("train_worker", os.path.join(config.SRC_ROOT, "workers", "train_worker.py"), True),
    ):
        if os.path.exists(path):
            line(label, OK)
        else:
            if required:
                failures += 1
            line(label, BAD if required else WARN, "not found")

    ggufs = [f for f in os.listdir(config.GGUF_DIR)] if os.path.isdir(config.GGUF_DIR) else []
    bases = [f for f in ggufs if f.endswith(".gguf") and "adapter" not in f.lower()]
    line("base GGUF", OK if bases else WARN,
         bases[0] if bases else "none — convert one from the Engine tab")

    for label, mod in (("core.store", "store"), ("core.pipeline", "pipeline"),
                       ("core.generate", "generate"), ("core.evaluate", "evaluate"),
                       ("core.voice", "voice"), ("core.feed", "feed")):
        check_import(label, mod)
except Exception as e:
    failures += 1
    line("config / .env", BAD, str(e)[:60])

print()
if failures:
    print(f"  {failures} problem(s) found. The app may still start, but the")
    print("  affected features will fail. See the Help tab for common fixes.")
else:
    print("  Everything checks out. Run: lora_env\\Scripts\\python src\\app.py")
print()
sys.exit(1 if failures else 0)
