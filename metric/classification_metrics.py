import numpy as np


# helpers ------------------------------------------------------------------

# Normalise les entrees de classification en tableaux NumPy
# pour simplifier les calculs vectorises dans tout le module.
def _as_arrays(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    return y_pred, y_true


# Force les scores probabilistes au bon format numerique
# avant les metriques comme le log loss, le Brier score ou l'AUC.
def _as_probability_arrays(y_proba, y_true):
    y_proba = np.asarray(y_proba, dtype=float)
    y_true = np.asarray(y_true)
    return y_proba, y_true


# Evite les divisions par zero et renvoie 0.0
# pour garder des metriques stables meme sur des cas degeneres.
def _safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


# Coupe les probabilites aux bornes ouvertes ]0, 1[
# pour eviter les log(0) dans le calcul du log loss.
def _clip_probabilities(y_proba):
    epsilon = np.finfo(float).eps
    return np.clip(y_proba, epsilon, 1 - epsilon)


# Centralise le calcul de TP, TN, FP et FN
# afin que toutes les metriques derivent de la meme logique.
def _confusion_counts(y_pred, y_true):
    y_pred, y_true = _as_arrays(y_pred, y_true)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, tn, fp, fn



# Confusion counts ----------------------------------------------------------

def count_true_positives(y_pred, y_true):
    """Calcule le nombre de positifs correctement detectes."""
    tp, _, _, _ = _confusion_counts(y_pred, y_true)
    return tp


def count_true_negatives(y_pred, y_true):
    """Calcule le nombre de negatifs correctement detectes."""
    _, tn, _, _ = _confusion_counts(y_pred, y_true)
    return tn


def count_false_positives(y_pred, y_true):
    """Calcule le nombre de faux positifs pour une classification binaire."""
    _, _, fp, _ = _confusion_counts(y_pred, y_true)
    return fp


def count_false_negatives(y_pred, y_true):
    """Calcule le nombre de faux negatifs pour une classification binaire."""
    _, _, _, fn = _confusion_counts(y_pred, y_true)
    return fn


# Classification metrics ----------------------------------------------------

def accuracy(y_pred, y_true):
    """Calcule la proportion de bonnes predictions."""
    tp, tn, fp, fn = _confusion_counts(y_pred, y_true)
    return _safe_divide(tp + tn, tp + tn + fp + fn)


def precision(y_pred, y_true):
    """Calcule la qualite des predictions positives."""
    tp, _, fp, _ = _confusion_counts(y_pred, y_true)
    return _safe_divide(tp, tp + fp)


def recall(y_pred, y_true):
    """Calcule la capacite a detecter les positifs."""
    tp, _, _, fn = _confusion_counts(y_pred, y_true)
    return _safe_divide(tp, tp + fn)


def specificity(y_pred, y_true):
    """Calcule la capacite a detecter les negatifs."""
    _, tn, fp, _ = _confusion_counts(y_pred, y_true)
    return _safe_divide(tn, tn + fp)


def f1_score(y_pred, y_true):
    """Calcule la moyenne harmonique entre precision et recall."""
    precision_value = precision(y_pred, y_true)
    recall_value = recall(y_pred, y_true)
    return _safe_divide(
        2 * precision_value * recall_value,
        precision_value + recall_value,
    )


def balanced_accuracy(y_pred, y_true):
    """Calcule la moyenne entre recall et specificity."""
    recall_value = recall(y_pred, y_true)
    specificity_value = specificity(y_pred, y_true)
    return (recall_value + specificity_value) / 2


def mcc(y_pred, y_true):
    """Calcule la correlation de Matthews."""
    tp, tn, fp, fn = _confusion_counts(y_pred, y_true)

    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return float(_safe_divide(numerator, denominator))


def predicted_positive_rate(y_pred, y_true):
    """Calcule la proportion de predictions positives."""
    tp, tn, fp, fn = _confusion_counts(y_pred, y_true)
    total = tp + tn + fp + fn
    return _safe_divide(tp + fp, total)


def prevalence(y_pred, y_true):
    """Calcule la proportion reelle de positifs."""
    tp, tn, fp, fn = _confusion_counts(y_pred, y_true)
    total = tp + tn + fp + fn
    return _safe_divide(tp + fn, total)


def recall_prevalence_ratio(y_pred, y_true):
    """Calcule le ratio entre le recall et la prevalence."""
    recall_value = recall(y_pred, y_true)
    prevalence_value = prevalence(y_pred, y_true)
    return _safe_divide(recall_value, prevalence_value)


# Probabilistic metrics -----------------------------------------------------

def log_loss(y_proba, y_true):
    """Calcule la qualite des probabilites predites."""
    y_proba, y_true = _as_probability_arrays(y_proba, y_true)
    y_proba = _clip_probabilities(y_proba)

    losses = y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba)
    return float(-np.mean(losses))


def baseline_log_loss(y_true):
    """Calcule le log loss de reference base sur la prevalence."""
    y_true = np.asarray(y_true)
    prevalence_value = float(np.mean(y_true))

    if prevalence_value in (0.0, 1.0):
        return 0.0

    return float(
        -(
            prevalence_value * np.log(prevalence_value)
            + (1 - prevalence_value) * np.log(1 - prevalence_value)
        )
    )


def brier_score(y_proba, y_true):
    """Calcule l'erreur quadratique moyenne des probabilites predites."""
    y_proba, y_true = _as_probability_arrays(y_proba, y_true)
    return float(np.mean((y_proba - y_true) ** 2))


# Ranking metrics -----------------------------------------------------------

def roc_auc(y_proba, y_true):
    """Calcule la probabilite qu'un positif ait un score superieur a un negatif."""
    y_proba, y_true = _as_probability_arrays(y_proba, y_true)

    positive_mask = y_true == 1
    negative_mask = y_true == 0
    n_positive = int(np.sum(positive_mask))
    n_negative = int(np.sum(negative_mask))

    if n_positive == 0 or n_negative == 0:
        return 0.0

    order = np.argsort(y_proba)
    sorted_scores = y_proba[order]
    ranks = np.empty(len(y_proba), dtype=float)

    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1

        average_rank = (start + 1 + end) / 2
        ranks[order[start:end]] = average_rank
        start = end

    positive_ranks_sum = float(np.sum(ranks[positive_mask]))
    mann_whitney_u = positive_ranks_sum - (n_positive * (n_positive + 1) / 2)

    return _safe_divide(mann_whitney_u, n_positive * n_negative)


def classification_metric_dict(y_pred, y_true):
    """Construit le bloc de metriques de classification pour une fenetre."""
    return {
        "tp": count_true_positives(y_pred, y_true),
        "tn": count_true_negatives(y_pred, y_true),
        "fp": count_false_positives(y_pred, y_true),
        "fn": count_false_negatives(y_pred, y_true),
        "precision": precision(y_pred, y_true),
        "recall": recall(y_pred, y_true),
        "specificity": specificity(y_pred, y_true),
        "mcc": mcc(y_pred, y_true),
        "ppr": predicted_positive_rate(y_pred, y_true),
        "prevalence": prevalence(y_pred, y_true),
    }


def probabilistic_metric_dict(y_proba, y_true, undefined_auc=np.nan):
    """Construit le bloc de metriques probabilistes pour une fenetre."""
    y_true = np.asarray(y_true)
    auc_value = float(undefined_auc)

    if np.unique(y_true).size == 2:
        auc_value = roc_auc(y_proba, y_true)

    return {
        "auc": auc_value,
        "logloss": log_loss(y_proba, y_true),
        "brier": brier_score(y_proba, y_true),
    }
