import numpy as np
import pandas as pd

from xgboost import XGBClassifier

import config
from data_gathering import prebuilt_dataset
from metric import scoring_function
from metric import window_metrics


def _validate_dataset_size(momentum, n_windows):
    """Verifie que le dataset contient assez de lignes pour le nombre de fenetres demande."""
    required = config.INITIAL_TRAIN_SIZE + n_windows * config.MODEL_LIFE_WINDOW
    if len(momentum) < required:
        raise ValueError(
            f"Dataset trop petit : {len(momentum)} lignes, il en faut au moins {required}."
        )


def load_backtest_dataset(n_windows=None, path=None):
    """Charge le dataset preconstruit et valide sa taille pour le backtest voulu."""
    if n_windows is None:
        n_windows = config.N_WINDOWS
    if path is None:
        path = config.DATASET_PATH

    dataset = prebuilt_dataset.load_prebuilt_dataset(path)
    _validate_dataset_size(dataset["momentum"], n_windows)
    return dataset


def train_model(X_train, y_train, params=None, seed=42):
    """Entraine un XGBoost sur le train courant avec les parametres fournis."""
    model = XGBClassifier(**config.build_xgb_params(seed, overrides=params))
    model.fit(X_train, y_train)
    return model


def run_window_backtest(
    params=None,
    n_windows=None,
    collect_predictions=False,
    dataset=None,
):
    """Execute le backtest sequentiel fenetre par fenetre et retourne les metriques produites."""
    if n_windows is None:
        n_windows = config.N_WINDOWS

    if dataset is None:
        dataset = load_backtest_dataset(n_windows)
    else:
        _validate_dataset_size(dataset["momentum"], n_windows)

    X = dataset["data"]
    y = dataset["momentum"]

    X_train = X[:config.INITIAL_TRAIN_SIZE]
    y_train = y[:config.INITIAL_TRAIN_SIZE]

    balance = config.INITIAL_BALANCE
    rows = []
    prediction_rows = []

    for window_id in range(1, n_windows + 1):
        window_start = config.INITIAL_TRAIN_SIZE + (window_id - 1) * config.MODEL_LIFE_WINDOW
        window_end = window_start + config.MODEL_LIFE_WINDOW

        X_window = X[window_start:window_end]
        y_window = y[window_start:window_end]

        model = train_model(
            X_train,
            y_train,
            params=params,
            seed=config.RANDOM_STATE + window_id,
        )
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_proba = model.predict_proba(X_window)[:, 1]

        metric_row = window_metrics.build_window_metric_dict(
            y_window,
            y_proba,
            balance_start=balance,
            threshold=config.THRESHOLD,
            tp_gain=config.TP_GAIN,
            fp_loss=config.FP_LOSS,
            history_rows=rows,
            y_train_true=y_train,
            y_train_proba=y_train_proba,
        )
        balance = metric_row["balance_end"]

        row = {
            "window_id": window_id,
            "train_size_before_update": len(X_train),
            "test_window_size": config.MODEL_LIFE_WINDOW,
        }
        row.update(metric_row)
        rows.append(row)

        if collect_predictions:
            for sample_offset, (true_value, proba_value) in enumerate(zip(y_window, y_proba)):
                prediction_rows.append(
                    {
                        "window_id": window_id,
                        "dataset_index": int(window_start + sample_offset),
                        "y_true": int(true_value),
                        "y_proba": float(proba_value),
                    }
                )

        X_train = np.vstack([X_train, X_window])
        y_train = np.concatenate([y_train, y_window])

    if collect_predictions:
        return rows, prediction_rows

    return rows


def summarize_rows(rows):
    """Agrege une liste de fenetres en un resume global de performance."""
    gain_realized = float(sum(row["gain_realized"] for row in rows))
    gain_max_possible = float(sum(row["gain_max_possible"] for row in rows))

    if abs(gain_max_possible) < 1e-12:
        gain_ratio = 0.0
    else:
        gain_ratio = gain_realized / gain_max_possible

    gain_effective_values = np.asarray(
        [row["gain_effective"] for row in rows],
        dtype=float,
    )

    return {
        "n_windows": int(len(rows)),
        "tp": int(sum(row["tp"] for row in rows)),
        "tn": int(sum(row["tn"] for row in rows)),
        "fp": int(sum(row["fp"] for row in rows)),
        "fn": int(sum(row["fn"] for row in rows)),
        "gain_realized": gain_realized,
        "gain_max_possible": gain_max_possible,
        "gain_ratio": float(gain_ratio),
        "gain_effective_mean": float(np.mean(gain_effective_values)) if gain_effective_values.size else 0.0,
    }


def evaluate_rows_with_score(rows, score_model):
    """Applique la score function a des fenetres et calcule les statistiques associees."""
    if not rows:
        return {
            "n_windows": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "gain_realized": 0.0,
            "gain_max_possible": 0.0,
            "gain_ratio": 0.0,
            "gain_effective_mean": 0.0,
            "score_median": 0.0,
            "score_mean": 0.0,
            "score_corr_gain_effective": 0.0,
            "score_corr_method": config.TARGET_CORR_METHOD,
        }

    score_df = pd.DataFrame(rows)
    score_values = scoring_function.apply_score_model_frame(score_df, score_model)
    target_gain = np.asarray([row["gain_effective"] for row in rows], dtype=float)

    summary = summarize_rows(rows)
    summary.update(
        {
            "score_median": float(np.median(score_values)),
            "score_mean": float(np.mean(score_values)),
            "score_corr_gain_effective": float(
                scoring_function.correlation_safe(
                    score_values,
                    target_gain,
                    method=config.TARGET_CORR_METHOD,
                )
            ),
            "score_corr_method": config.TARGET_CORR_METHOD,
        }
    )
    return summary
