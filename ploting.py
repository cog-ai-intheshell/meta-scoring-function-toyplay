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


def plot_referential_score_evolution(
    referential_df,
    title: str = "Trajectoire du modele dans le referentiel 4D",
    path: str | Path | None = None,
    split_window_id: int | None = None,
) -> Optional[Path]:
    """Trace la trajectoire 4D du modele avec 3 axes spatiaux et la 4e dimension codee par couleur."""
    required_columns = [
        "model_life_window",
        "classification",
        "separation",
        "generalization",
        "stabilite",
        "score_metric",
    ]

    missing_columns = [
        name for name in required_columns if name not in referential_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Colonnes manquantes pour le referentiel: "
            + ", ".join(missing_columns)
        )

    x = np.asarray(referential_df["classification"], dtype=float)
    y = np.asarray(referential_df["separation"], dtype=float)
    z = np.asarray(referential_df["generalization"], dtype=float)
    stability = np.asarray(referential_df["stabilite"], dtype=float)
    windows = np.asarray(referential_df["model_life_window"], dtype=int)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Thin path showing temporal evolution along model-life-window order.
    ax.plot(x, y, z, color="#adb5bd", linewidth=1.2, alpha=0.8)

    scatter = ax.scatter(
        x,
        y,
        z,
        c=stability,
        cmap="viridis",
        s=36,
        alpha=0.95,
    )

    ax.scatter(
        [x[0]],
        [y[0]],
        [z[0]],
        color="#2a9d8f",
        s=90,
        marker="o",
        label=f"depart (window {windows[0]})",
    )
    ax.scatter(
        [x[-1]],
        [y[-1]],
        [z[-1]],
        color="#c1121f",
        s=110,
        marker="X",
        label=f"arrivee (window {windows[-1]})",
    )
    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        color="black",
        s=80,
        marker="+",
        label="origine O",
    )

    if split_window_id is not None:
        split_mask = windows == int(split_window_id)
        if np.any(split_mask):
            split_index = int(np.argmax(split_mask))
            ax.scatter(
                [x[split_index]],
                [y[split_index]],
                [z[split_index]],
                color="#f77f00",
                s=100,
                marker="D",
                label=f"frontiere train/test ({split_window_id})",
            )

    ax.set_title(
        title + "\nAxes spatiaux: classification, separation, generalization | Couleur: stabilite"
    )
    ax.set_xlabel("classification")
    ax.set_ylabel("separation")
    ax.set_zlabel("generalization")
    ax.view_init(elev=22, azim=35)
    ax.legend(loc="upper left")

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.08)
    colorbar.set_label("stabilite")

    fig.tight_layout()

    if path is not None:
        output_path = _safe_output_path(path)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    plt.show()
    plt.close(fig)
    return None

