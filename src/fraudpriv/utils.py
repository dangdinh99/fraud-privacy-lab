"""Reproducibility and config helpers."""

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    """Write a dict as indented JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_yaml(obj: dict[str, Any], path: str | Path) -> None:
    """Write a dict as YAML, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def env_info() -> dict[str, Any]:
    """Python and package versions, git commit, and whether the working tree is dirty."""
    import platform
    import subprocess
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "torch", "opacus", "pyyaml"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def configure_device(device: str) -> None:
    """Prepare reproducible training on ``device`` ("cpu" or "cuda").

    For CUDA: requires a GPU, sets the cuBLAS workspace needed for deterministic matrix
    products, and enables deterministic algorithms (an operation without a deterministic
    GPU version then raises instead of silently varying between runs).
    """
    if device == "cpu":
        return
    import torch

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(f"device {device!r} requested but CUDA is not available")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
