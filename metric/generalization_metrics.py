import numpy as np


# helpers ------------------------------------------------------------------

# Normalise des suites numeriques de metrics calculees par fold.
def _as_float_array(values):
    return np.asarray(values, dtype=float)


# Evite les divisions par zero pour les agregations.
def _safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


# Centralise les ecarts-types avec un comportement stable sur tableaux vides.
def _std_or_zero(values):
    values = _as_float_array(values)
    if values.size == 0:
        return 0.0
    return float(np.std(values))


# Bias / variance diagnostics ----------------------------------------------

def empirical_bias(validation_losses):
    """Approxime le biais par la moyenne des losses de validation."""
    validation_losses = _as_float_array(validation_losses)
    if validation_losses.size == 0:
        return 0.0
    return float(np.mean(validation_losses))


def empirical_variance(validation_losses):
    """Approxime la variance par la variance des losses de validation."""
    validation_losses = _as_float_array(validation_losses)
    if validation_losses.size == 0:
        return 0.0
    return float(np.var(validation_losses))


def empirical_residual_noise(y_true, y_score):
    """Approxime le bruit residuel par la variance empirique des residus (sigma^2)."""
    y_true = _as_float_array(y_true)
    y_score = _as_float_array(y_score)

    if y_true.shape != y_score.shape:
        raise ValueError("y_true et y_score doivent avoir la meme forme")

    if y_true.size == 0:
        return 0.0

    residuals = y_true - y_score
    return float(np.var(residuals))


def empirical_overfitting(train_losses, validation_losses):
    """Approxime l'overfitting par l'ecart moyen validation - train."""
    train_losses = _as_float_array(train_losses)
    validation_losses = _as_float_array(validation_losses)

    if train_losses.size == 0 or validation_losses.size == 0:
        return 0.0

    if train_losses.shape != validation_losses.shape:
        raise ValueError("train_losses et validation_losses doivent avoir la meme forme")

    return float(np.mean(validation_losses) - np.mean(train_losses))


# Fold-level robustness -----------------------------------------------------

def stability_score(values):
    """Score simple de stabilite, proche de 1 quand la dispersion est faible."""
    values = _as_float_array(values)
    if values.size == 0:
        return 0.0
    return float(_safe_divide(1.0, 1.0 + np.std(values)))


def worst_fold(values, higher_is_better=True):
    """Retourne le pire score observe sur les folds."""
    values = _as_float_array(values)
    if values.size == 0:
        return 0.0
    if higher_is_better:
        return float(np.min(values))
    return float(np.max(values))


def robustness_gap(values, higher_is_better=True):
    """Mesure l'ecart entre la moyenne et le pire fold."""
    values = _as_float_array(values)
    if values.size == 0:
        return 0.0

    mean_value = float(np.mean(values))
    worst_value = worst_fold(values, higher_is_better=higher_is_better)

    if higher_is_better:
        return mean_value - worst_value
    return worst_value - mean_value


# Fold dispersion metrics ---------------------------------------------------

def recall_std(recall_values):
    """Ecart-type des recalls par fold."""
    return _std_or_zero(recall_values)


def precision_std(precision_values):
    """Ecart-type des precisions par fold."""
    return _std_or_zero(precision_values)


def ppr_std(ppr_values):
    """Ecart-type des predicted positive rates par fold."""
    return _std_or_zero(ppr_values)


def logloss_std(logloss_values):
    """Ecart-type des log loss par fold."""
    return _std_or_zero(logloss_values)


def auc_std(auc_values):
    """Ecart-type des AUCs par fold."""
    return _std_or_zero(auc_values)


def brier_std(brier_values):
    """Ecart-type des Brier scores par fold."""
    return _std_or_zero(brier_values)


def sharpe_std(sharpe_values):
    """Ecart-type des ratios de Sharpe par fold."""
    return _std_or_zero(sharpe_values)


def worst_sharpe(sharpe_values):
    """Pire ratio de Sharpe observe sur les folds."""
    sharpe_values = _as_float_array(sharpe_values)
    if sharpe_values.size == 0:
        return 0.0
    return float(np.min(sharpe_values))


def max_drawdown_worst(drawdown_values):
    """Plus forte drawdown observee sur les folds."""
    drawdown_values = _as_float_array(drawdown_values)
    if drawdown_values.size == 0:
        return 0.0
    return float(np.max(drawdown_values))


def threshold_std(threshold_values):
    """Ecart-type des seuils optimaux observes par fold."""
    return _std_or_zero(threshold_values)


def threshold_drift(threshold_values):
    """Amplitude de derive des seuils, egale a max(th) - min(th)."""
    threshold_values = _as_float_array(threshold_values)
    if threshold_values.size == 0:
        return 0.0
    return float(np.max(threshold_values) - np.min(threshold_values))


def interfold_variance(values):
    """Variance inter-fold d'une suite de scores."""
    values = _as_float_array(values)
    if values.size == 0:
        return 0.0
    return float(np.var(values))


def interfold_std(values):
    """Ecart-type inter-fold d'une suite de scores."""
    return _std_or_zero(values)
