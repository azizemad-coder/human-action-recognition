import os
import sys
from pathlib import Path


def _ensure_src_on_path():
    repo_root = Path(__file__).resolve().parent
    # When running in Colab, cwd is typically the repo root; this is safe elsewhere too
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_ensure_src_on_path()


