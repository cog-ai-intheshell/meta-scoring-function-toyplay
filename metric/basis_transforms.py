import numpy as np

from metric import algebra


def _as_matrix(matrix):
    """Convertit une entree en matrice 2D flottante ou leve une erreur explicite."""
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("L'operation n'est definie que pour une matrice 2D.")

    return matrix


def orthogonalize_family_qr(matrix, tol=None):
    """Orthogonalise une famille libre rangee en colonnes via une decomposition QR."""
    matrix = _as_matrix(matrix)
    rank = algebra.family_rank(matrix, tol=tol)

    if rank != matrix.shape[1]:
        raise ValueError(
            "La famille doit etre libre pour etre orthogonalisee sans perte de dimension."
        )

    orthogonal_matrix, r_matrix = np.linalg.qr(matrix, mode="reduced")
    diagonal_signs = np.sign(np.diag(r_matrix))
    diagonal_signs[diagonal_signs == 0.0] = 1.0
    sign_matrix = np.diag(diagonal_signs)

    orthogonal_matrix = orthogonal_matrix @ sign_matrix
    r_matrix = sign_matrix @ r_matrix
    transform_matrix = np.linalg.solve(
        r_matrix,
        np.eye(r_matrix.shape[0], dtype=float),
    )

    return {
        "orthogonal_matrix": orthogonal_matrix,
        "r_matrix": r_matrix,
        "transform_matrix": transform_matrix,
    }


def normalize_family_columns(matrix, eps=1e-12):
    """Normalise les vecteurs-colonnes d'une famille pour leur donner une norme 1."""
    matrix = _as_matrix(matrix)
    norms = np.linalg.norm(matrix, axis=0)

    if np.any(norms <= float(eps)):
        raise ValueError(
            "Impossible de normaliser une famille contenant un vecteur de norme nulle."
        )

    transform_matrix = np.diag(1.0 / norms)
    normalized_matrix = matrix @ transform_matrix

    return {
        "normalized_matrix": normalized_matrix,
        "norms_before": norms,
        "transform_matrix": transform_matrix,
    }
