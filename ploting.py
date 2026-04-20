from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _safe_output_path(path: str | Path) -> Path:
    """Prepare le chemin de sortie d'une figure en creant le dossier si besoin."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_probability_predictions(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    threshold: float,
    title: str = "Momentum predictions vs threshold",
    path: str | Path | None = None,
) -> Optional[Path]:
    """Trace les probabilites predites et distingue TP, TN, FP et FN autour du seuil."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)

    if y_true_arr.shape[0] != y_proba_arr.shape[0]:
        raise ValueError("y_true et y_proba doivent avoir la meme longueur")

    y_pred = (y_proba_arr >= threshold).astype(int)
    x = np.arange(len(y_true_arr))

    fig = plt.figure(figsize=(16, 5))

    plt.scatter(x, y_proba_arr, c=y_true_arr, cmap="bwr", alpha=0.25, s=10)

    tp = (y_true_arr == 1) & (y_pred == 1)
    tn = (y_true_arr == 0) & (y_pred == 0)
    fp = (y_true_arr == 0) & (y_pred == 1)
    fn = (y_true_arr == 1) & (y_pred == 0)

    plt.scatter(x[tp], y_proba_arr[tp], color="green", s=35, label="TP")
    plt.scatter(x[tn], y_proba_arr[tn], color="blue", s=35, label="TN")
    plt.scatter(x[fp], y_proba_arr[fp], color="red", s=50, label="FP")
    plt.scatter(x[fn], y_proba_arr[fn], color="orange", s=50, label="FN")

    plt.axhline(threshold, linestyle="--", linewidth=1.5)

    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Probabilite predite")
    plt.legend()
    plt.tight_layout()

    if path is not None:
        output_path = _safe_output_path(path)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    plt.show()
    plt.close(fig)
    return None
