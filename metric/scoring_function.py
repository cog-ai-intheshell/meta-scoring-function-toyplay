import json
from pathlib import Path

import numpy as np
import pandas as pd

from metric import basis_transforms
from metric import correlation_analysis


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
    return correlation_safe(x, y, method="pearson", eps=eps)


def correlation_safe(x, y, method="pearson", eps=1e-12):
    """Correlation robuste configurable avec exclusion des NaN."""
    return correlation_analysis.correlation_safe(x, y, method=method, eps=eps)


def _fit_marginal_corr_weights(Z, target_z, corr_method="pearson", eps=1e-12):
    """Construit un operateur lineaire a partir des correlations marginales a la cible."""
    w = np.array(
        [
            correlation_safe(Z[:, j], target_z, method=corr_method, eps=eps)
            for j in range(Z.shape[1])
        ],
        dtype=float,
    )

    w_norm = np.linalg.norm(w)
    if w_norm < eps:
        return np.ones_like(w) / np.sqrt(len(w))

    return w / w_norm


def _fit_ridge_weights(Z, target_z, ridge_lambda):
    """Resout l'operateur ridge (Z^T Z + lambda I)^-1 Z^T y sur donnees standardisees."""
    ridge_lambda = float(ridge_lambda)
    if ridge_lambda < 0.0:
        raise ValueError("ridge_lambda doit etre positif ou nul.")

    n_features = Z.shape[1]
    lhs = Z.T @ Z + ridge_lambda * np.eye(n_features, dtype=float)
    rhs = Z.T @ target_z

    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def _orthogonalize_basis_coordinates(Z, method, eps=1e-12):
    """Orthogonalise une base standardisee et retourne la transformation associee."""
    if method != "qr":
        raise ValueError(
            f"Methode d'orthogonalisation inconnue: {method}"
        )

    orthogonalization = basis_transforms.orthogonalize_family_qr(Z, tol=eps)
    return orthogonalization["orthogonal_matrix"], {
        "enabled": True,
        "method": method,
        "transform_matrix": orthogonalization["transform_matrix"].tolist(),
        "r_matrix": orthogonalization["r_matrix"].tolist(),
    }


def _normalize_basis_coordinates(Z, eps=1e-12):
    """Normalise les vecteurs d'une base et retourne la transformation associee."""
    normalization = basis_transforms.normalize_family_columns(Z, eps=eps)
    return normalization["normalized_matrix"], {
        "enabled": True,
        "norms_before": normalization["norms_before"].tolist(),
        "transform_matrix": normalization["transform_matrix"].tolist(),
    }


def fit_linear_score_model(
    X,
    feature_names,
    target,
    target_name="target_gain",
    eps=1e-12,
    operator_type="ridge",
    ridge_lambda=1.0,
    orthogonalize=False,
    orthogonalization_method="qr",
    normalize=False,
    target_corr_method="pearson",
):
    """Apprend un score lineaire a partir d'une base de metriques standardisees."""
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

    learning_basis = Z
    orthogonalization_payload = {
        "enabled": False,
        "method": None,
        "transform_matrix": None,
        "r_matrix": None,
    }
    normalization_payload = {
        "enabled": False,
        "norms_before": None,
        "transform_matrix": None,
    }
    basis_transform = np.eye(Z.shape[1], dtype=float)

    if orthogonalize:
        learning_basis, orthogonalization_payload = _orthogonalize_basis_coordinates(
            Z,
            method=orthogonalization_method,
            eps=eps,
        )
        basis_transform = basis_transform @ np.asarray(
            orthogonalization_payload["transform_matrix"],
            dtype=float,
        )

    if normalize:
        learning_basis, normalization_payload = _normalize_basis_coordinates(
            learning_basis,
            eps=eps,
        )
        basis_transform = basis_transform @ np.asarray(
            normalization_payload["transform_matrix"],
            dtype=float,
        )

    if operator_type == "ridge":
        w = _fit_ridge_weights(learning_basis, target_z, ridge_lambda=ridge_lambda)
    elif operator_type == "marginal_corr":
        w = _fit_marginal_corr_weights(
            learning_basis,
            target_z,
            corr_method=target_corr_method,
            eps=eps,
        )
    else:
        raise ValueError(f"Type d'operateur lineaire inconnu: {operator_type}")

    score_values = score_operator(learning_basis, w)

    return {
        "model_type": "linear",
        "operator_type": operator_type,
        "feature_names": kept_feature_names,
        "mu": mu_kept.tolist(),
        "sigma": sigma_kept.tolist(),
        "weights": w.tolist(),
        "ridge_lambda": float(ridge_lambda),
        "orthogonalized": bool(orthogonalization_payload["enabled"]),
        "orthogonalization_method": orthogonalization_payload["method"],
        "orthogonalization_transform": orthogonalization_payload["transform_matrix"],
        "orthogonalization_r": orthogonalization_payload["r_matrix"],
        "normalized": bool(normalization_payload["enabled"]),
        "normalization_norms_before": normalization_payload["norms_before"],
        "normalization_transform": normalization_payload["transform_matrix"],
        "basis_transform": basis_transform.tolist(),
        "target_corr_method": target_corr_method,
        "target_name": target_name,
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "training_target_corr": correlation_safe(
            score_values,
            target,
            method=target_corr_method,
            eps=eps,
        ),
    }, score_values


def apply_linear_score_model(X, score_model):
    """Applique un score lineaire sauvegarde a une matrice deja alignee sur feature_names."""
    Z = score_coordinates_with_model(X, score_model)
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


def extract_score_coordinates_frame(frame_like, score_model):
    """Extrait les coordonnees Z dans la base finale juste avant l'operateur lineaire."""
    model_type = score_model.get("model_type", "linear")

    if model_type == "linear":
        frame = _as_feature_frame(frame_like)
        feature_names = score_model["feature_names"]
        X = frame[feature_names].values.astype(float)
        Z = score_coordinates_with_model(X, score_model)
        return pd.DataFrame(Z, columns=feature_names)

    if model_type != "hierarchical_family_score":
        raise ValueError(f"Type de modele de score inconnu: {model_type}")

    frame = _as_feature_frame(frame_like)
    family_components = {}

    for family_name, family_model in score_model["family_models"].items():
        family_components[family_name] = _apply_linear_score_model_from_frame(
            frame,
            family_model,
        )

    final_model = score_model["final_model"]
    family_score_names = final_model["feature_names"]
    family_score_frame = pd.DataFrame(
        {
            name: family_components[name]
            for name in family_score_names
        }
    )
    Z = score_coordinates_with_model(
        family_score_frame[family_score_names].values,
        final_model,
    )
    return pd.DataFrame(Z, columns=family_score_names)


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


def transform_score_coordinates(Z, score_model):
    """Projette des coordonnees standardisees dans la base eventuellement orthogonalisee."""
    Z = np.asarray(Z, dtype=float)

    if score_model.get("basis_transform") is not None:
        transform_matrix = np.asarray(
            score_model["basis_transform"],
            dtype=float,
        )
        if transform_matrix.ndim != 2:
            raise ValueError("La matrice de transformation du modele est invalide.")
        if Z.shape[1] != transform_matrix.shape[0]:
            raise ValueError(
                "Le nombre de colonnes de Z ne correspond pas a la transformation du modele."
            )
        return Z @ transform_matrix

    if not score_model.get("orthogonalized", False):
        return Z

    transform_matrix = np.asarray(
        score_model["orthogonalization_transform"],
        dtype=float,
    )

    if transform_matrix.ndim != 2:
        raise ValueError("La matrice d'orthogonalisation du modele est invalide.")

    if Z.shape[1] != transform_matrix.shape[0]:
        raise ValueError(
            "Le nombre de colonnes de Z ne correspond pas a la transformation du modele."
        )

    return Z @ transform_matrix


def score_coordinates_with_model(X, score_model):
    """Construit les coordonnees finales dans la base du modele avant l'operateur lineaire."""
    Z = standardize_with_score_model(X, score_model)
    return transform_score_coordinates(Z, score_model)


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
