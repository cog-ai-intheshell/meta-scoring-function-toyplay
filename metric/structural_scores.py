import numpy as np


# helpers ------------------------------------------------------------------

def _safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def clip01(value):
    """Borne une valeur dans l'intervalle [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))


def weighted_geometric_mean(values, weights):
    """Moyenne geometrique ponderee generique."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if values.shape != weights.shape:
        raise ValueError("values et weights doivent avoir la meme forme")

    if values.size == 0 or np.sum(weights) <= 0:
        return 0.0

    if np.any(values < 0):
        raise ValueError("weighted_geometric_mean attend des valeurs non negatives")

    normalized_weights = weights / np.sum(weights)
    powered_values = np.power(values, normalized_weights)

    return float(np.prod(powered_values))


def weighted_geometric_mean_2(x1, x2, a, b):
    return weighted_geometric_mean([x1, x2], [a, b])


def weighted_geometric_mean_3(x1, x2, x3, a, b, c):
    return weighted_geometric_mean([x1, x2, x3], [a, b, c])


def weighted_geometric_mean_4(x1, x2, x3, x4, a, b, c, d):
    return weighted_geometric_mean([x1, x2, x3, x4], [a, b, c, d])


# Separation ---------------------------------------------------------------

def auc_norm(auc_value):
    return clip01(_safe_divide(auc_value - 0.5, 0.5))


def threshold_confidence_ratio_norm(ratio_value):
    return clip01(_safe_divide(ratio_value - 1.0, 1.5))


def threshold_confidence_gap_norm(gap_value):
    return clip01(_safe_divide(gap_value, 0.12))


def margin_quality(threshold_confidence_ratio_value, threshold_confidence_gap_value):
    ratio_norm_value = threshold_confidence_ratio_norm(threshold_confidence_ratio_value)
    gap_norm_value = threshold_confidence_gap_norm(threshold_confidence_gap_value)

    return weighted_geometric_mean_2(
        ratio_norm_value,
        gap_norm_value,
        0.55,
        0.45,
    )


def separation_raw_score(
    auc_value,
    threshold_confidence_ratio_value,
    threshold_confidence_gap_value,
):
    return weighted_geometric_mean_2(
        auc_norm(auc_value),
        margin_quality(
            threshold_confidence_ratio_value,
            threshold_confidence_gap_value,
        ),
        0.75,
        0.25,
    )


# Calibration --------------------------------------------------------------

def brier_baseline(prevalence_value):
    prevalence_value = clip01(prevalence_value)
    return float(prevalence_value * (1.0 - prevalence_value))


def relative_logloss_gain(log_loss_value, baseline_log_loss_value):
    return _safe_divide(baseline_log_loss_value - log_loss_value, baseline_log_loss_value)


def logloss_gain_norm(log_loss_value, baseline_log_loss_value):
    relative_gain = relative_logloss_gain(log_loss_value, baseline_log_loss_value)
    return clip01(_safe_divide(relative_gain, 0.20))


def logloss_abs_norm(log_loss_value, lower=0.45, upper=0.69):
    return clip01(_safe_divide(upper - log_loss_value, upper - lower))


def logloss_norm(log_loss_value, baseline_log_loss_value):
    return weighted_geometric_mean_2(
        logloss_gain_norm(log_loss_value, baseline_log_loss_value),
        logloss_abs_norm(log_loss_value),
        0.50,
        0.50,
    )


def brier_gain_norm(brier_score_value, brier_baseline_value):
    relative_gain = _safe_divide(brier_baseline_value - brier_score_value, brier_baseline_value)
    return clip01(_safe_divide(relative_gain, 0.25))


def calibration_raw_score(log_loss_value, baseline_log_loss_value, brier_score_value, prevalence_value):
    brier_baseline_value = brier_baseline(prevalence_value)

    return weighted_geometric_mean_2(
        logloss_norm(log_loss_value, baseline_log_loss_value),
        brier_gain_norm(brier_score_value, brier_baseline_value),
        0.60,
        0.40,
    )


# Decision -----------------------------------------------------------------

def signal_quality(precision_value, recall_value, specificity_value):
    return weighted_geometric_mean_3(
        clip01(precision_value),
        clip01(recall_value),
        clip01(specificity_value),
        0.40,
        0.35,
        0.25,
    )


def threshold_alignment(threshold_value, implicit_threshold_value):
    return clip01(1.0 - abs(threshold_value - implicit_threshold_value))


def positive_rate_alignment(predicted_positive_rate_value, prevalence_value):
    return clip01(1.0 - abs(predicted_positive_rate_value - prevalence_value))


def threshold_coherence(threshold_value, implicit_threshold_value, predicted_positive_rate_value, prevalence_value):
    return weighted_geometric_mean_2(
        threshold_alignment(threshold_value, implicit_threshold_value),
        positive_rate_alignment(predicted_positive_rate_value, prevalence_value),
        0.55,
        0.45,
    )


def decision_raw_score(
    precision_value,
    recall_value,
    specificity_value,
    threshold_value,
    implicit_threshold_value,
    predicted_positive_rate_value,
    prevalence_value,
):
    return weighted_geometric_mean_2(
        signal_quality(precision_value, recall_value, specificity_value),
        threshold_coherence(
            threshold_value,
            implicit_threshold_value,
            predicted_positive_rate_value,
            prevalence_value,
        ),
        0.70,
        0.30,
    )


# Generalization -----------------------------------------------------------

def overfit_norm(overfitting_value):
    return clip01(1.0 - _safe_divide(overfitting_value, 0.20))


def variance_norm(variance_value):
    return clip01(1.0 - _safe_divide(variance_value, 0.03))


def auc_stability_norm(auc_std_value):
    return clip01(1.0 - _safe_divide(auc_std_value, 0.20))


def bias_norm(bias_value, low=0.55, high=0.85):
    return clip01(_safe_divide(high - bias_value, high - low))


def generalization_raw_score(overfitting_value, variance_value, auc_std_value, bias_value):
    return weighted_geometric_mean_4(
        overfit_norm(overfitting_value),
        variance_norm(variance_value),
        auc_stability_norm(auc_std_value),
        bias_norm(bias_value),
        0.35,
        0.25,
        0.20,
        0.20,
    )
