import numpy as np

from metric import classification_metrics
from metric import decision_metrics
from metric import gain_metrics
from metric import generalization_metrics


CLASSIFICATION_COLS = [
    "tp",
    "tn",
    "fp",
    "fn",
    "precision",
    "recall",
    "specificity",
    "mcc",
    "balanced_accuracy",
]

SEPARATION_COLS = [
    "auc",
    "logloss",
    "brier",
    "median_dist_tp",
    "median_dist_tn",
    "median_dist_fp",
    "median_dist_fn",
    "threshold_confidence_ratio",
    "threshold_confidence_gap",
]

GENERALIZATION_COLS = [
    "empirical_overfitting",
    "empirical_bias",
    "empirical_variance",
    "empirical_residual_noise",
]

STABILITY_COLS = [
    "auc_std",
    "logloss_std",
    "recall_std",
    "precision_std",
    "ppr_std",
    "threshold_std",
    "threshold_drift",
    "robustness_gap",
    "stability_score",
]

GAIN_COLS = [
    "gain_realized",
    "gain_max_possible",
    "gain_effective",
    "balance_start",
    "balance_end",
]

REALIZED_WINDOW_METRIC_COLS = (
    CLASSIFICATION_COLS
    + SEPARATION_COLS
    + GENERALIZATION_COLS
    + STABILITY_COLS
)

ALL_METRIC_COLS = REALIZED_WINDOW_METRIC_COLS + GAIN_COLS


def _finite_history_values(history_rows, key):
    """Extrait une suite historique finie pour une metrique donnee."""
    return np.asarray(
        [
            float(row[key])
            for row in history_rows
            if key in row and np.isfinite(row[key])
        ],
        dtype=float,
    )


def _build_generalization_metric_dict(y_true, y_proba, history_rows):
    """Construit le bloc de generalization a partir de l'historique et de la fenetre courante."""
    validation_losses = _finite_history_values(history_rows, "logloss")
    train_losses = _finite_history_values(history_rows, "train_logloss")

    return {
        "empirical_overfitting": generalization_metrics.empirical_overfitting(
            train_losses,
            validation_losses,
        ),
        "empirical_bias": generalization_metrics.empirical_bias(validation_losses),
        "empirical_variance": generalization_metrics.empirical_variance(validation_losses),
        "empirical_residual_noise": generalization_metrics.empirical_residual_noise(
            y_true,
            y_proba,
        ),
    }


def _build_stability_metric_dict(history_rows):
    """Construit le bloc de stabilite a partir de l'historique cumule des fenetres."""
    auc_values = _finite_history_values(history_rows, "auc")
    logloss_values = _finite_history_values(history_rows, "logloss")
    recall_values = _finite_history_values(history_rows, "recall")
    precision_values = _finite_history_values(history_rows, "precision")
    ppr_values = _finite_history_values(history_rows, "ppr")
    threshold_values = _finite_history_values(history_rows, "implicit_threshold")
    mcc_values = _finite_history_values(history_rows, "mcc")

    return {
        "auc_std": generalization_metrics.auc_std(auc_values),
        "logloss_std": generalization_metrics.logloss_std(logloss_values),
        "recall_std": generalization_metrics.recall_std(recall_values),
        "precision_std": generalization_metrics.precision_std(precision_values),
        "ppr_std": generalization_metrics.ppr_std(ppr_values),
        "threshold_std": generalization_metrics.threshold_std(threshold_values),
        "threshold_drift": generalization_metrics.threshold_drift(threshold_values),
        "robustness_gap": generalization_metrics.robustness_gap(
            mcc_values,
            higher_is_better=True,
        ),
        "stability_score": generalization_metrics.stability_score(mcc_values),
    }


def build_window_metric_dict(
    y_true,
    y_proba,
    balance_start,
    threshold,
    tp_gain,
    fp_loss,
    history_rows=None,
    y_train_true=None,
    y_train_proba=None,
):
    """Construit toutes les metriques utiles pour une fenetre."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)
    history_rows = [] if history_rows is None else list(history_rows)

    y_pred = decision_metrics.decision_label(y_proba, threshold=threshold)

    metrics = {}
    metrics.update(classification_metrics.classification_metric_dict(y_pred, y_true))
    metrics.update(classification_metrics.probabilistic_metric_dict(y_proba, y_true))
    metrics.update(
        decision_metrics.threshold_geometry_metric_dict(
            y_proba,
            y_true,
            threshold=threshold,
        )
    )

    if y_train_true is None or y_train_proba is None:
        train_logloss = metrics["logloss"]
    else:
        train_logloss = classification_metrics.log_loss(y_train_proba, y_train_true)

    metrics["train_logloss"] = float(train_logloss)

    history_plus_current = history_rows + [metrics]
    metrics.update(
        _build_generalization_metric_dict(
            y_true,
            y_proba,
            history_plus_current,
        )
    )
    metrics.update(_build_stability_metric_dict(history_plus_current))
    metrics.update(
        gain_metrics.balance_metric_dict(
            y_pred,
            y_true,
            balance_start=balance_start,
            tp_gain=tp_gain,
            fp_loss=fp_loss,
        )
    )

    return metrics
