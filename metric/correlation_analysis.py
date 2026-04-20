import numpy as np
import pandas as pd


def pearson_corr_ignore_nan(x, y, eps=1e-12):
    """Correlation de Pearson en ignorant les paires contenant un NaN."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid_mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid_mask) < 2:
        return float("nan")

    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    if np.std(x_valid) < eps or np.std(y_valid) < eps:
        return float("nan")

    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def correlation_matrix_ignore_nan(X, column_names, eps=1e-12):
    """Matrice de correlation construite paire par paire sans imputer les NaN."""
    X = np.asarray(X, dtype=float)
    n_cols = X.shape[1]
    corr = np.full((n_cols, n_cols), np.nan, dtype=float)

    for i in range(n_cols):
        corr[i, i] = 1.0
        for j in range(i + 1, n_cols):
            corr_value = pearson_corr_ignore_nan(X[:, i], X[:, j], eps=eps)
            corr[i, j] = corr_value
            corr[j, i] = corr_value

    return pd.DataFrame(corr, index=column_names, columns=column_names)


def extract_correlated_pairs(corr_df, threshold=0.98):
    """Retourne toutes les paires triees par correlation absolue decroissante."""
    threshold = float(threshold)
    pairs = []
    column_names = list(corr_df.columns)

    for i in range(len(column_names)):
        for j in range(i + 1, len(column_names)):
            signed_corr = float(corr_df.iloc[i, j])
            if not np.isfinite(signed_corr):
                continue

            abs_corr = abs(signed_corr)
            pairs.append(
                {
                    "left": column_names[i],
                    "right": column_names[j],
                    "corr": signed_corr,
                    "abs_corr": abs_corr,
                    "is_above_threshold": abs_corr >= threshold,
                }
            )

    pairs.sort(key=lambda item: (-item["abs_corr"], item["left"], item["right"]))
    return pairs


def select_uncorrelated_columns_by_target_corr(corr_df, target_corr, threshold=0.98):
    """Garde prioritairement les colonnes les plus corrélées à la cible."""
    threshold = float(threshold)
    target_corr = {
        name: abs(float(value)) if np.isfinite(value) else 0.0
        for name, value in target_corr.items()
    }

    sorted_columns = sorted(
        corr_df.columns,
        key=lambda name: (-target_corr.get(name, 0.0), name),
    )

    kept_columns = []
    dropped_columns = []

    for column in sorted_columns:
        correlated_with = None

        for kept in kept_columns:
            corr_value = float(corr_df.loc[kept, column])
            if np.isfinite(corr_value) and abs(corr_value) >= threshold:
                correlated_with = {
                    "dropped": column,
                    "kept": kept,
                    "corr": corr_value,
                    "abs_corr": abs(corr_value),
                    "dropped_target_corr": target_corr.get(column, 0.0),
                    "kept_target_corr": target_corr.get(kept, 0.0),
                }
                break

        if correlated_with is None:
            kept_columns.append(column)
        else:
            dropped_columns.append(correlated_with)

    return kept_columns, dropped_columns
