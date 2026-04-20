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
