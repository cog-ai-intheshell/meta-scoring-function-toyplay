import numpy as np

from metric import classification_metrics
from metric import decision_metrics
from metric import gain_metrics


CLASSIFICATION_COLS = [
    "tp",
    "tn",
    "fp",
    "fn",
    "precision",
    "recall",
    "specificity",
    "mcc",
]

SEPARATION_COLS = [
    "auc",
]

CALIBRATION_COLS = [
    "logloss",
    "brier",
]

DECISION_COLS = [
    "ppr",
    "prevalence",
]

THRESHOLD_GEOMETRY_COLS = [
    "dist_tp",
    "dist_fp",
    "dist_tn",
    "dist_fn",
]

GAIN_COLS = [
    "gain_realized",
    "gain_max_possible",
    "gain_effective",
    "balance_start",
    "balance_end",
]

EX_ANTE_METRIC_COLS = (
    CLASSIFICATION_COLS
    + SEPARATION_COLS
    + CALIBRATION_COLS
    + DECISION_COLS
    + THRESHOLD_GEOMETRY_COLS
)

ALL_METRIC_COLS = EX_ANTE_METRIC_COLS + GAIN_COLS


def build_window_metric_dict(y_true, y_proba, balance_start, threshold, tp_gain, fp_loss):
    """Construit toutes les metriques utiles pour une fenetre."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

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
