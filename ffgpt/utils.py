from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_states() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_states(states: dict[str, Any]) -> None:
    if not states:
        return
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.random.set_rng_state(states["torch"])
    if torch.cuda.is_available() and states.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)
    return out_path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def latest_checkpoint(checkpoint_dir: str | Path, mode: str) -> Path | None:
    ckpt_dir = Path(checkpoint_dir)
    matches = sorted(ckpt_dir.glob(f"{mode}_step*.pt"))
    if not matches:
        return None
    return matches[-1]


def compute_confusion_matrix(targets: list[int], predictions: list[int], num_classes: int = 19) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, predictions):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def _import_matplotlib() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


def plot_curve(
    steps: list[int],
    values: list[float],
    output_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, values)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_multi_curve(
    steps: list[int],
    series: dict[str, list[float]],
    output_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, values in series.items():
        if len(values) == len(steps):
            ax.plot(steps, values, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, output_path: str | Path, title: str) -> None:
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted Sum")
    ax.set_ylabel("True Sum")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
