import json
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


def plot_referential_score_evolution_html_3d(
    referential_df,
    title: str = "Trajectoire interactive du modele dans le referentiel 3D",
    path: str | Path | None = None,
    split_window_id: int | None = None,
) -> Optional[Path]:
    """Genere un HTML interactif 3D fusionnant les fenetres dev et holdout."""
    required_columns = [
        "model_life_window",
        "classification",
        "separation",
        "generalization",
        "stabilite",
        "score_metric",
        "gain_effective",
    ]

    missing_columns = [
        name for name in required_columns if name not in referential_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Colonnes manquantes pour le referentiel interactif: "
            + ", ".join(missing_columns)
        )

    if path is None:
        raise ValueError("Un chemin de sortie HTML est requis pour le referentiel interactif.")

    output_path = _safe_output_path(path)

    records = []
    for row in referential_df[required_columns].itertuples(index=False):
        window_id = int(row.model_life_window)
        split_label = "full"
        if split_window_id is not None:
            split_label = "dev" if window_id <= int(split_window_id) else "holdout"

        records.append(
            {
                "window_id": window_id,
                "classification": float(row.classification),
                "separation": float(row.separation),
                "generalization": float(row.generalization),
                "stabilite": float(row.stabilite),
                "score_metric": float(row.score_metric),
                "gain_effective": float(row.gain_effective),
                "split": split_label,
            }
        )

    payload = {
        "title": title,
        "records": records,
        "split_window_id": None if split_window_id is None else int(split_window_id),
    }

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f6f7f9;
      color: #111827;
    }}
    .wrap {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 20px;
    }}
    #chart {{
      width: 100%;
      height: 85vh;
      background: white;
      border-radius: 12px;
      box-shadow: 0 6px 24px rgba(0, 0, 0, 0.08);
    }}
    .meta {{
      margin: 0 0 12px;
      color: #4b5563;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <p class="meta">Axes spatiaux: classification, separation, generalization | Couleur: split dev/holdout | Hover: stabilite, score, gain</p>
    <div id="chart"></div>
  </div>
  <script>
    const payload = {json.dumps(payload, ensure_ascii=False)};
    const rows = payload.records;

    const xs = rows.map((row) => row.classification);
    const ys = rows.map((row) => row.separation);
    const zs = rows.map((row) => row.generalization);

    const devRows = rows.filter((row) => row.split === "dev");
    const holdoutRows = rows.filter((row) => row.split === "holdout");

    const hoverTemplate =
      "window=%{{customdata[0]}}<br>" +
      "split=%{{customdata[1]}}<br>" +
      "stabilite=%{{customdata[2]:.4f}}<br>" +
      "score=%{{customdata[3]:.4f}}<br>" +
      "gain_effective=%{{customdata[4]:.4f}}<extra></extra>";

    function markerTrace(name, subset, color, symbol, size) {{
      return {{
        type: "scatter3d",
        mode: "markers",
        name,
        x: subset.map((row) => row.classification),
        y: subset.map((row) => row.separation),
        z: subset.map((row) => row.generalization),
        customdata: subset.map((row) => [
          row.window_id,
          row.split,
          row.stabilite,
          row.score_metric,
          row.gain_effective,
        ]),
        hovertemplate: hoverTemplate,
        marker: {{
          size,
          color,
          symbol,
          opacity: 0.92,
          line: {{
            width: 0.5,
            color: "rgba(255,255,255,0.6)"
          }}
        }}
      }};
    }}

    const traces = [
      {{
        type: "scatter3d",
        mode: "lines",
        name: "trajectoire complete",
        x: xs,
        y: ys,
        z: zs,
        line: {{
          color: "#adb5bd",
          width: 4
        }},
        hoverinfo: "skip"
      }},
      markerTrace("dev", devRows, "#2a9d8f", "circle", 5),
      markerTrace("holdout", holdoutRows, "#c1121f", "diamond", 6),
      {{
        type: "scatter3d",
        mode: "markers",
        name: "depart",
        x: [xs[0]],
        y: [ys[0]],
        z: [zs[0]],
        marker: {{ size: 9, color: "#1d4ed8", symbol: "circle-open" }},
        hovertemplate: "depart window=%{{text}}<extra></extra>",
        text: [rows[0].window_id]
      }},
      {{
        type: "scatter3d",
        mode: "markers",
        name: "arrivee",
        x: [xs[xs.length - 1]],
        y: [ys[ys.length - 1]],
        z: [zs[zs.length - 1]],
        marker: {{ size: 10, color: "#7c2d12", symbol: "x" }},
        hovertemplate: "arrivee window=%{{text}}<extra></extra>",
        text: [rows[rows.length - 1].window_id]
      }},
      {{
        type: "scatter3d",
        mode: "markers",
        name: "origine O",
        x: [0],
        y: [0],
        z: [0],
        marker: {{ size: 7, color: "black", symbol: "cross" }},
        hovertemplate: "origine O<extra></extra>"
      }}
    ];

    if (payload.split_window_id !== null) {{
      const splitRow = rows.find((row) => row.window_id === payload.split_window_id);
      if (splitRow) {{
        traces.push({{
          type: "scatter3d",
          mode: "markers",
          name: `frontiere train/test (${{payload.split_window_id}})`,
          x: [splitRow.classification],
          y: [splitRow.separation],
          z: [splitRow.generalization],
          marker: {{ size: 9, color: "#f77f00", symbol: "diamond-open" }},
          hovertemplate: "frontiere window=%{{text}}<extra></extra>",
          text: [splitRow.window_id]
        }});
      }}
    }}

    const layout = {{
      title: {{ text: payload.title }},
      paper_bgcolor: "#f6f7f9",
      plot_bgcolor: "white",
      legend: {{
        orientation: "h",
        yanchor: "bottom",
        y: 1.02,
        xanchor: "left",
        x: 0
      }},
      margin: {{ l: 0, r: 0, b: 0, t: 60 }},
      scene: {{
        xaxis: {{ title: "classification" }},
        yaxis: {{ title: "separation" }},
        zaxis: {{ title: "generalization" }},
        camera: {{
          eye: {{ x: 1.55, y: 1.4, z: 0.95 }}
        }}
      }}
    }};

    Plotly.newPlot("chart", traces, layout, {{
      responsive: true,
      displaylogo: false
    }});
  </script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    return output_path
