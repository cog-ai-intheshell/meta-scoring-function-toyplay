import numpy as np


def _as_matrix(matrix):
    """Convertit une entree en matrice 2D flottante ou leve une erreur explicite."""
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("L'operation n'est definie que pour une matrice 2D.")

    return matrix


def _as_square_matrix(matrix):
    """Convertit une entree en matrice carree flottante ou leve une erreur explicite."""
    matrix = _as_matrix(matrix)

    n_rows, n_cols = matrix.shape
    if n_rows != n_cols:
        raise ValueError("L'operation n'est definie que pour une matrice carree.")

    return matrix


def matrix_determinant(matrix):
    """Calcule le determinant d'une matrice carree en precision flottante."""
    matrix = _as_square_matrix(matrix)
    return float(np.linalg.det(matrix))


def family_rank(matrix, tol=None):
    """Calcule le rang numerique d'une famille de vecteurs rangee en colonnes."""
    matrix = _as_matrix(matrix)
    singular_values = np.linalg.svd(matrix, compute_uv=False)

    if singular_values.size == 0:
        return 0

    if tol is None:
        tol = (
            max(matrix.shape)
            * np.finfo(float).eps
            * float(np.max(singular_values))
        )

    return int(np.sum(singular_values > float(tol)))


def is_free_family(matrix, tol=None):
    """Indique si les vecteurs-colonnes forment une famille libre."""
    matrix = _as_matrix(matrix)
    return family_rank(matrix, tol=tol) == matrix.shape[1]


def is_generating_family(matrix, ambient_dimension=None, tol=None):
    """Indique si les vecteurs-colonnes engendrent l'espace ambiant choisi."""
    matrix = _as_matrix(matrix)

    if ambient_dimension is None:
        ambient_dimension = matrix.shape[0]

    ambient_dimension = int(ambient_dimension)
    if ambient_dimension <= 0:
        raise ValueError("ambient_dimension doit etre strictement positif.")

    return family_rank(matrix, tol=tol) == ambient_dimension


def orthogonality_analysis(matrix, tol=1e-10):
    """Mesure l'orthogonalite d'une famille via les cosinus hors diagonale."""
    matrix = _as_matrix(matrix)
    gram = gram_matrix(matrix)
    norms = np.sqrt(np.clip(np.diag(gram), a_min=0.0, a_max=None))

    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = np.outer(norms, norms)
        cosine_matrix = np.divide(
            gram,
            denominator,
            out=np.zeros_like(gram, dtype=float),
            where=denominator > 0.0,
        )

    np.fill_diagonal(cosine_matrix, 0.0)
    max_abs_cosine_off_diagonal = (
        float(np.max(np.abs(cosine_matrix)))
        if cosine_matrix.size
        else 0.0
    )

    return {
        "is_orthogonal": bool(max_abs_cosine_off_diagonal <= float(tol)),
        "max_abs_cosine_off_diagonal": max_abs_cosine_off_diagonal,
        "cosine_matrix": cosine_matrix,
        "tolerance": float(tol),
    }


def unit_norm_analysis(matrix, tol=1e-10):
    """Analyse si les vecteurs-colonnes sont tous de norme 1 a la tolerance pres."""
    matrix = _as_matrix(matrix)
    norms = np.linalg.norm(matrix, axis=0)

    if norms.size == 0:
        max_abs_norm_deviation = 0.0
    else:
        max_abs_norm_deviation = float(np.max(np.abs(norms - 1.0)))

    return {
        "is_unit_norm_family": bool(max_abs_norm_deviation <= float(tol)),
        "norms": norms,
        "max_abs_norm_deviation": max_abs_norm_deviation,
        "tolerance": float(tol),
    }


def gram_matrix(matrix):
    """Construit la matrice de Gram X^T X associee a une matrice de donnees."""
    matrix = _as_matrix(matrix)
    return matrix.T @ matrix


def covariance_matrix(matrix, ddof=1):
    """Calcule la covariance colonne par colonne et garantit une sortie 2D."""
    matrix = _as_matrix(matrix)
    cov = np.asarray(np.cov(matrix, rowvar=False, ddof=ddof), dtype=float)

    if cov.ndim == 0:
        cov = cov.reshape(1, 1)

    return cov


def regularized_logdet(matrix, ridge_lambda=1e-6):
    """Calcule log(det(M + lambda I)) avec une regularisation diagonale."""
    matrix = _as_square_matrix(matrix)
    ridge_lambda = float(ridge_lambda)

    if ridge_lambda < 0.0:
        raise ValueError("ridge_lambda doit etre positif ou nul.")

    regularized = matrix + ridge_lambda * np.eye(matrix.shape[0], dtype=float)
    sign, logdet = np.linalg.slogdet(regularized)

    if sign <= 0:
        return float("-inf")

    return float(logdet)
