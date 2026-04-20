import numpy as np


# helpers ------------------------------------------------------------------

# Normalise les scores de decision en tableau numerique.
def _as_score_array(y_score):
    """Convertit un vecteur de scores en tableau NumPy flottant."""
    return np.asarray(y_score, dtype=float)


# Normalise a la fois les scores et la verite terrain.
def _as_score_and_true_arrays(y_score, y_true):
    """Convertit conjointement les scores et les labels en tableaux compatibles."""
    y_score = _as_score_array(y_score)
    y_true = np.asarray(y_true)
    return y_score, y_true


# Evite les divisions par zero pour les metriques agregees.
def _safe_divide(numerator, denominator):
    """Effectue une division sure en renvoyant 0.0 si le denominateur est nul."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


# Convertit un score en decision binaire a partir d'un seuil.
def _decision_label_array(y_score, threshold):
    """Transforme un score probabiliste en prediction binaire a un seuil donne."""
    y_score = _as_score_array(y_score)
    return (y_score >= threshold).astype(int)


# Construit un ensemble stable de seuils candidats a partir des scores.
def _threshold_candidates(y_score):
    """Construit un ensemble de seuils candidats stables a partir des scores observes."""
    y_score = np.clip(_as_score_array(y_score), 0.0, 1.0)

    if y_score.size == 0:
        return np.asarray([0.5], dtype=float)

    return np.unique(np.concatenate(([0.0, 0.5, 1.0], y_score)))


# Mesure la distance absolue au seuil, interpretee comme marge de decision.
def _threshold_distance_array(y_score, threshold):
    """Calcule la distance absolue de chaque score au seuil de decision."""
    y_score = _as_score_array(y_score)
    return np.abs(y_score - threshold)


# Calcule une moyenne sur un sous-ensemble; renvoie 0.0 si le masque est vide.
def _masked_mean(values, mask):
    """Calcule une moyenne sur un sous-ensemble masque avec sortie sure si vide."""
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if values.shape != mask.shape:
        raise ValueError("values et mask doivent avoir la meme forme")

    if not np.any(mask):
        return 0.0

    return float(np.mean(values[mask]))


def _masked_median(values, mask):
    """Calcule une mediane sur un sous-ensemble masque avec sortie sure si vide."""
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if values.shape != mask.shape:
        raise ValueError("values et mask doivent avoir la meme forme")

    if not np.any(mask):
        return 0.0

    return float(np.median(values[mask]))


# Calcule rapidement un F1 binaire sans dependre d'un autre module.
def _f1_at_threshold(y_score, y_true, threshold):
    """Calcule le F1 obtenu si l'on applique un seuil donne aux probabilites."""
    y_score, y_true = _as_score_and_true_arrays(y_score, y_true)
    y_pred = _decision_label_array(y_score, threshold)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision_value = _safe_divide(tp, tp + fp)
    recall_value = _safe_divide(tp, tp + fn)

    return _safe_divide(
        2 * precision_value * recall_value,
        precision_value + recall_value,
    )


# Decision outputs ---------------------------------------------------------

def decision_label(y_score, threshold):
    """Retourne le label de decision binaire pour chaque score."""
    return _decision_label_array(y_score, threshold)


def decision_score(y_score):
    """Retourne les scores de decision tels quels."""
    return _as_score_array(y_score)


def decision_confidence(y_score, threshold):
    """Retourne la confiance comme distance absolue au seuil."""
    return _threshold_distance_array(y_score, threshold)


def decision_warnings(y_score, threshold, warning_margin=0.05):
    """Signale les decisions trop proches du seuil ou hors intervalle [0, 1]."""
    y_score = _as_score_array(y_score)
    distances = _threshold_distance_array(y_score, threshold)

    return np.where(
        (y_score < 0) | (y_score > 1),
        "score_out_of_range",
        np.where(distances < warning_margin, "close_to_threshold", "none"),
    )


# Threshold-based metrics --------------------------------------------------

def threshold_distance_mean(y_score, threshold):
    """Distance moyenne au seuil."""
    distances = _threshold_distance_array(y_score, threshold)
    return float(np.mean(distances))


def threshold_distance_correct(y_score, y_true, threshold):
    """Distance moyenne au seuil pour les decisions correctes."""
    y_score, y_true = _as_score_and_true_arrays(y_score, y_true)
    y_pred = _decision_label_array(y_score, threshold)
    distances = _threshold_distance_array(y_score, threshold)
    return _masked_mean(distances, y_pred == y_true)


def threshold_distance_wrong(y_score, y_true, threshold):
    """Distance moyenne au seuil pour les decisions incorrectes."""
    y_score, y_true = _as_score_and_true_arrays(y_score, y_true)
    y_pred = _decision_label_array(y_score, threshold)
    distances = _threshold_distance_array(y_score, threshold)
    return _masked_mean(distances, y_pred != y_true)


def threshold_confidence_ratio(y_score, y_true, threshold):
    """Rapport entre la confiance moyenne correcte et incorrecte."""
    correct_distance = threshold_distance_correct(y_score, y_true, threshold)
    wrong_distance = threshold_distance_wrong(y_score, y_true, threshold)
    return _safe_divide(correct_distance, wrong_distance)


def threshold_confidence_gap(y_score, y_true, threshold):
    """Ecart de confiance moyen entre decisions correctes et incorrectes."""
    correct_distance = threshold_distance_correct(y_score, y_true, threshold)
    wrong_distance = threshold_distance_wrong(y_score, y_true, threshold)
    return correct_distance - wrong_distance


def threshold_distance_tp(y_score, y_true, threshold):
    """Distance moyenne au seuil pour les vrais positifs."""
    y_score, y_true = _as_score_and_true_arrays(y_score, y_true)
    y_pred = _decision_label_array(y_score, threshold)
    distances = _threshold_distance_array(y_score, threshold)
    mask = (y_true == 1) & (y_pred == 1)
    return _masked_mean(distances, mask)


def threshold_distance_fp(y_score, y_true, threshold):
    """Distance moyenne au seuil pour les faux positifs."""
    y_score, y_true = _as_score_and_true_arrays(y_score, y_true)
    y_pred = _decision_label_array(y_score, threshold)
    distances = _threshold_distance_array(y_score, threshold)
    mask = (y_true == 0) & (y_pred == 1)
    return _masked_mean(distances, mask)


def threshold_distance_fn(y_score, y_true, threshold):
    """Distance moyenne au seuil pour les faux negatifs."""
    y_score, y_true = _as_score_and_true_arrays(y_score, y_true)
    y_pred = _decision_label_array(y_score, threshold)
    distances = _threshold_distance_array(y_score, threshold)
    mask = (y_true == 1) & (y_pred == 0)
    return _masked_mean(distances, mask)


def threshold_distance_tn(y_score, y_true, threshold):
    """Distance moyenne au seuil pour les vrais negatifs."""
    y_score, y_true = _as_score_and_true_arrays(y_score, y_true)
    y_pred = _decision_label_array(y_score, threshold)
    distances = _threshold_distance_array(y_score, threshold)
    mask = (y_true == 0) & (y_pred == 0)
    return _masked_mean(distances, mask)


def threshold_distance_median_tp(y_score, y_true, threshold):
    """Distance mediane au seuil pour les vrais positifs."""
    y_score, y_true = _as_score_and_true_arrays(y_score, y_true)
    y_pred = _decision_label_array(y_score, threshold)
    distances = _threshold_distance_array(y_score, threshold)
    mask = (y_true == 1) & (y_pred == 1)
    return _masked_median(distances, mask)


def threshold_distance_median_fp(y_score, y_true, threshold):
    """Distance mediane au seuil pour les faux positifs."""
    y_score, y_true = _as_score_and_true_arrays(y_score, y_true)
    y_pred = _decision_label_array(y_score, threshold)
    distances = _threshold_distance_array(y_score, threshold)
    mask = (y_true == 0) & (y_pred == 1)
    return _masked_median(distances, mask)


def threshold_distance_median_fn(y_score, y_true, threshold):
    """Distance mediane au seuil pour les faux negatifs."""
    y_score, y_true = _as_score_and_true_arrays(y_score, y_true)
    y_pred = _decision_label_array(y_score, threshold)
    distances = _threshold_distance_array(y_score, threshold)
    mask = (y_true == 1) & (y_pred == 0)
    return _masked_median(distances, mask)


def threshold_distance_median_tn(y_score, y_true, threshold):
    """Distance mediane au seuil pour les vrais negatifs."""
    y_score, y_true = _as_score_and_true_arrays(y_score, y_true)
    y_pred = _decision_label_array(y_score, threshold)
    distances = _threshold_distance_array(y_score, threshold)
    mask = (y_true == 0) & (y_pred == 0)
    return _masked_median(distances, mask)


def implicit_linear_threshold(y_score, y_true):
    """Seuil implicite qui aligne au mieux le taux de positifs predits sur la prevalence."""
    y_score, y_true = _as_score_and_true_arrays(y_score, y_true)

    if y_score.size == 0:
        return 0.5

    prevalence_value = float(np.mean(y_true))
    candidate_thresholds = _threshold_candidates(y_score)
    predicted_positive_rates = np.asarray(
        [np.mean(y_score >= threshold) for threshold in candidate_thresholds],
        dtype=float,
    )

    distances = np.abs(predicted_positive_rates - prevalence_value)
    tie_break = np.abs(candidate_thresholds - 0.5)
    best_index = np.lexsort((tie_break, distances))[0]

    return float(candidate_thresholds[best_index])


def top_threshold_candidates(y_score, y_true, top_k=5, default_threshold=0.5, scoring_fn=None):
    """Retourne les meilleurs seuils candidats selon une fonction de score."""
    y_score, y_true = _as_score_and_true_arrays(y_score, y_true)

    if y_score.size == 0 or top_k <= 0:
        return np.asarray([], dtype=float)

    candidate_thresholds = _threshold_candidates(y_score)

    if scoring_fn is None:
        scored_thresholds = [
            (
                _f1_at_threshold(y_score, y_true, threshold),
                abs(threshold - default_threshold),
                threshold,
            )
            for threshold in candidate_thresholds
        ]
    else:
        scored_thresholds = [
            (
                float(scoring_fn(threshold, y_score, y_true)),
                abs(threshold - default_threshold),
                threshold,
            )
            for threshold in candidate_thresholds
        ]

    scored_thresholds.sort(key=lambda item: (-item[0], item[1], item[2]))
    top_thresholds = [threshold for _, _, threshold in scored_thresholds[:top_k]]

    return np.asarray(top_thresholds, dtype=float)


def select_best_threshold(y_score, y_true, default_threshold=0.5, scoring_fn=None):
    """Selectionne le meilleur seuil candidat."""
    candidates = top_threshold_candidates(
        y_score,
        y_true,
        top_k=1,
        default_threshold=default_threshold,
        scoring_fn=scoring_fn,
    )

    if candidates.size == 0:
        return float(default_threshold)

    return float(candidates[0])


def threshold_geometry_metric_dict(y_score, y_true, threshold):
    """Construit le bloc des metriques de separation autour du seuil."""
    return {
        "median_dist_tp": threshold_distance_median_tp(y_score, y_true, threshold),
        "median_dist_tn": threshold_distance_median_tn(y_score, y_true, threshold),
        "median_dist_fp": threshold_distance_median_fp(y_score, y_true, threshold),
        "median_dist_fn": threshold_distance_median_fn(y_score, y_true, threshold),
        "threshold_confidence_ratio": threshold_confidence_ratio(
            y_score,
            y_true,
            threshold,
        ),
        "threshold_confidence_gap": threshold_confidence_gap(
            y_score,
            y_true,
            threshold,
        ),
        "implicit_threshold": implicit_linear_threshold(y_score, y_true),
    }
