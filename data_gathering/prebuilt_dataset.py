from pathlib import Path

import numpy as np
import pandas as pd

import config


def _run_lengths(binary_values):
    """Calcule les longueurs de runs consecutifs pour les valeurs 0 et 1."""
    binary_values = np.asarray(binary_values, dtype=int)
    if binary_values.size == 0:
        return {0: [], 1: []}

    lengths = {0: [], 1: []}
    current_value = int(binary_values[0])
    current_length = 1

    for value in binary_values[1:]:
        value = int(value)
        if value == current_value:
            current_length += 1
            continue

        lengths[current_value].append(current_length)
        current_value = value
        current_length = 1

    lengths[current_value].append(current_length)
    return lengths


def sequence_statistics(momentum_labels):
    """Produit quelques statistiques descriptives sur la sequence temporelle du dataset."""
    momentum_labels = np.asarray(momentum_labels, dtype=int)
    run_lengths = _run_lengths(momentum_labels)
    switches = int(np.sum(momentum_labels[1:] != momentum_labels[:-1])) if momentum_labels.size > 1 else 0

    return {
        "n_samples": int(momentum_labels.size),
        "momentum_rate": float(np.mean(momentum_labels)) if momentum_labels.size else 0.0,
        "switches": switches,
        "avg_run_momentum": float(np.mean(run_lengths[1])) if run_lengths[1] else 0.0,
        "avg_run_zero": float(np.mean(run_lengths[0])) if run_lengths[0] else 0.0,
        "max_run_momentum": int(max(run_lengths[1])) if run_lengths[1] else 0,
        "max_run_zero": int(max(run_lengths[0])) if run_lengths[0] else 0,
    }


def load_prebuilt_dataset(path=config.DATASET_PATH):
    """Charge le dataset CSV final et retourne ses features, labels et statistiques de sequence."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable: {path}. Regenerer le CSV preconstruit avant execution."
        )

    df = pd.read_csv(path)
    feature_names = [
        name for name in df.columns if name.startswith(config.FEATURE_PREFIX)
    ]
    if not feature_names:
        raise ValueError(
            f"Aucune feature detectee dans {path}. "
            f"Les colonnes doivent commencer par '{config.FEATURE_PREFIX}'."
        )

    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Colonne cible absente dans {path}: '{config.TARGET_COLUMN}'."
        )

    y = df[config.TARGET_COLUMN].values.astype(int)

    return {
        "data": df[feature_names].values.astype(float),
        "momentum": y,
        "sequence_stats": sequence_statistics(y),
        "feature_names": feature_names,
        "path": path,
        "mode": "csv_prebuilt",
    }


def protocol_split_index(dev_windows=config.N_WINDOWS):
    """Retourne l'index de coupe entre dev et holdout dans le protocole complet."""
    return config.required_samples(dev_windows)


def load_protocol_dataset(
    dev_path=config.DATASET_DEV_PATH,
    holdout_path=config.DATASET_HOLDOUT_PATH,
):
    """Recharge le protocole complet en concaténant les CSV dev et holdout."""
    dev_dataset = load_prebuilt_dataset(dev_path)
    holdout_dataset = load_prebuilt_dataset(holdout_path)

    if dev_dataset["feature_names"] != holdout_dataset["feature_names"]:
        raise ValueError(
            "Les features du CSV dev et du CSV holdout ne correspondent pas."
        )

    X = np.vstack([dev_dataset["data"], holdout_dataset["data"]])
    y = np.concatenate([dev_dataset["momentum"], holdout_dataset["momentum"]])

    return {
        "data": X,
        "momentum": y,
        "sequence_stats": sequence_statistics(y),
        "feature_names": dev_dataset["feature_names"],
        "path": (
            f"{Path(dev_path).resolve()} + {Path(holdout_path).resolve()}"
        ),
        "mode": "csv_dev_plus_holdout",
    }
