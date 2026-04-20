import json
from pathlib import Path

import numpy as np
import pandas as pd


def standardize_matrix(X, eps=1e-12):
    """Centre-reduit une matrice en ignorant les NaN puis remplit les NaN restants par 0."""
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)

    keep_mask = np.isfinite(mu) & np.isfinite(sigma) & (sigma > eps)
    X_kept = X[:, keep_mask]
    mu_kept = mu[keep_mask]
    sigma_kept = sigma[keep_mask]

    Z = (X_kept - mu_kept) / sigma_kept
    Z = np.nan_to_num(Z, nan=0.0)

    return Z, keep_mask, mu_kept, sigma_kept


def score_operator(Z, w):
    """Applique l'operateur lineaire de score a une matrice deja standardisee."""
    Z = np.asarray(Z, dtype=float)
    w = np.asarray(w, dtype=float)
    return Z @ w


def pearson_corr_safe(x, y, eps=1e-12):
    """Correlation de Pearson robuste avec exclusion des NaN."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid_mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid_mask) < 2:
        return 0.0

    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    if np.std(x_valid) < eps or np.std(y_valid) < eps:
        return 0.0

    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def fit_linear_score_model(X, feature_names, target, target_name="target_gain", eps=1e-12):
    """Apprend un score lineaire a partir d'une base de metriques."""
    X = np.asarray(X, dtype=float)
    target = np.asarray(target, dtype=float)

    Z, keep_mask, mu_kept, sigma_kept = standardize_matrix(X, eps=eps)
    kept_feature_names = [
        name for name, keep in zip(feature_names, keep_mask) if keep
    ]

    if len(kept_feature_names) == 0:
        raise ValueError("Aucune feature exploitable pour apprendre la fonction de score")

    target_mean = np.nanmean(target)
    target_std = np.nanstd(target)
    if not np.isfinite(target_std) or target_std < eps:
        raise ValueError(f"La cible {target_name} est quasi constante : optimisation impossible.")

    target_z = (target - target_mean) / target_std
    target_z = np.nan_to_num(target_z, nan=0.0)

    w = np.array(
        [pearson_corr_safe(Z[:, j], target_z, eps=eps) for j in range(Z.shape[1])],
        dtype=float,
    )

    w_norm = np.linalg.norm(w)
    if w_norm < eps:
        w = np.ones_like(w) / np.sqrt(len(w))
    else:
        w = w / w_norm

    score_values = score_operator(Z, w)

    return {
        "model_type": "linear",
        "feature_names": kept_feature_names,
        "mu": mu_kept.tolist(),
        "sigma": sigma_kept.tolist(),
        "weights": w.tolist(),
        "target_name": target_name,
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "training_target_corr": pearson_corr_safe(score_values, target, eps=eps),
    }, score_values


def apply_linear_score_model(X, score_model):
    """Applique un score lineaire sauvegarde a une matrice deja alignee sur feature_names."""
    Z = standardize_with_score_model(X, score_model)
    weights = np.asarray(score_model["weights"], dtype=float)
    return score_operator(Z, weights)


def _as_feature_frame(frame_like):
    """Convertit une entree tabulaire quelconque en DataFrame pandas."""
    if isinstance(frame_like, pd.DataFrame):
        return frame_like.copy()
    return pd.DataFrame(frame_like)


def _apply_linear_score_model_from_frame(frame_like, score_model):
    """Aligne un DataFrame sur les features du modele puis applique le score lineaire."""
    frame = _as_feature_frame(frame_like)
    X = frame[score_model["feature_names"]].values.astype(float)
    return apply_linear_score_model(X, score_model)


def apply_score_model_frame(frame_like, score_model, return_components=False):
    """Applique un modele lineaire ou hierarchique a un DataFrame de metriques."""
    model_type = score_model.get("model_type", "linear")

    if model_type == "linear":
        score_values = _apply_linear_score_model_from_frame(frame_like, score_model)
        if return_components:
            return score_values, {}
        return score_values

    if model_type != "hierarchical_family_score":
        raise ValueError(f"Type de modele de score inconnu: {model_type}")

    frame = _as_feature_frame(frame_like)
    family_components = {}

    for family_name, family_model in score_model["family_models"].items():
        family_components[family_name] = _apply_linear_score_model_from_frame(
            frame,
            family_model,
        )

    family_score_names = score_model["final_model"]["feature_names"]
    family_score_frame = pd.DataFrame(
        {
            name: family_components[name]
            for name in family_score_names
        }
    )
    score_values = _apply_linear_score_model_from_frame(
        family_score_frame,
        score_model["final_model"],
    )

    if return_components:
        return score_values, family_components
    return score_values


def standardize_with_score_model(X, score_model):
    """Centre-reduit une matrice a l'aide des statistiques du modele de score."""
    X = np.asarray(X, dtype=float)
    mu = np.asarray(score_model["mu"], dtype=float)
    sigma = np.asarray(score_model["sigma"], dtype=float)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.shape[1] != len(mu):
        raise ValueError("Le nombre de colonnes de X ne correspond pas au modele de score")

    Z = (X - mu) / sigma
    Z = np.nan_to_num(Z, nan=0.0)
    return Z


def save_score_model(score_model, path, metadata=None):
    """Sauvegarde la fonction de score sur disque au format JSON."""
    payload = dict(score_model)
    if metadata is not None:
        payload["metadata"] = metadata

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_score_model(path):
    """Recharge une fonction de score sauvegardee."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))
