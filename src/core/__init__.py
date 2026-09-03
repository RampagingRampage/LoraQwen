"""core — the LoraQwen application package.

Importing `core` puts the project root on sys.path so the sibling modules
inside this package can keep importing each other by bare name (`import
store`, `import pipeline`) exactly as they did when they lived at the root,
without every one of them needing its own sys.path shim.
"""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)          # src/
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
