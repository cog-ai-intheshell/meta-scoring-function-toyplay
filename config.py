from pathlib import Path


RANDOM_STATE = 42

DATA_GATHERING_DIR = Path("data_gathering")
DIGITS_SOURCE_CSV_PATH = DATA_GATHERING_DIR / "digits_source.csv"
REFERENCE_CSV_PATH = DATA_GATHERING_DIR / "reference.csv"
FULL_DATASET_PATH = DATA_GATHERING_DIR / "dataset.csv"
DATASET_DEV_PATH = DATA_GATHERING_DIR / "dataset_dev.csv"
DATASET_HOLDOUT_PATH = DATA_GATHERING_DIR / "dataset_holdout.csv"
DATASET_PATH = DATASET_DEV_PATH
FEATURE_PREFIX = "pixel_"
TARGET_COLUMN = "momentum"
DIGIT_POSITIVE_MIN_LABEL = 5

INITIAL_TRAIN_SIZE = 97  # 1797 - 170 * 10 = 97
MODEL_LIFE_WINDOW = 10
FULL_PROTOCOL_WINDOWS = 170
N_WINDOWS = 120
SCORE_TRAIN_WINDOWS = N_WINDOWS
SCORE_TEST_WINDOWS = 0
OPTUNA_TRAIN_WINDOWS = N_WINDOWS
HOLDOUT_WINDOWS = FULL_PROTOCOL_WINDOWS - OPTUNA_TRAIN_WINDOWS

INITIAL_BALANCE = 10_000.0
TP_GAIN = 0.04
FP_LOSS = 0.03
THRESHOLD = 0.5
CORR_THRESHOLD = 0.90
GEOMETRY_RIDGE_LAMBDA = 1e-3
CORRELATION_METHODS = ("pearson", "spearman")
TARGET_CORR_METHOD = "spearman"  # "pearson" | "spearman"
REDUNDANCY_CORR_METHOD = "spearman"  # "pearson" | "spearman"
REQUIRE_FREE_FAMILY = True  # Doit rester fixe a True : la base finale doit toujours etre une famille libre.
REQUIRE_GENERATING_SUBSPACE = True  # Doit rester fixe a True : la base finale doit toujours engendrer son sous-espace.
ORTHOGONALIZE_FAMILY = False # Optionnel : utile quand on veut imposer une base finale orthogonale pour l'etude geometrique.
NORMALIZE_FAMILY = False # Optionnel : utile quand on veut imposer des vecteurs unitaires pour comparer directement les axes du referentiel.
SCORE_OPERATOR_TYPE = "marginal_corr" # "marginal_corr" ou "ridge"
SCORE_OPERATOR_RIDGE_LAMBDA = 1.0 # utilise seulement si SCORE_OPERATOR_TYPE = "ridge"

SCORE_MODEL_PATH = Path("artifacts/score_function.json")
BEST_PARAMS_PATH = Path("artifacts/optuna_best_xgb.json")
HOLDOUT_EVALUATION_PATH = Path("artifacts/holdout_evaluation.json")
HOLDOUT_PLOT_PATH = Path("artifacts/holdout_probability_predictions.png")
REFERENTIAL_PLOT_PATH = Path("artifacts/referential_score_evolution.png")
HOLDOUT_REFERENTIAL_PLOT_PATH = Path("artifacts/holdout_referential_score_evolution.png")
HOLDOUT_MODEL_RUNS_CSV_PATH = Path("artifacts/holdout_model_runs.csv")

GENERATOR_DIGITS_SOURCE_PATH = DIGITS_SOURCE_CSV_PATH
GENERATOR_REFERENCE_PATH = REFERENCE_CSV_PATH
GENERATOR_OUTPUT_PATH = FULL_DATASET_PATH
GENERATOR_DEV_OUTPUT_PATH = DATASET_DEV_PATH
GENERATOR_HOLDOUT_OUTPUT_PATH = DATASET_HOLDOUT_PATH
GENERATOR_RANDOM_STATE = RANDOM_STATE

OPTUNA_N_TRIALS = 20
OPTUNA_N_ESTIMATORS_BOUNDS = (50, 250)
OPTUNA_MAX_DEPTH_BOUNDS = (2, 8)
OPTUNA_LEARNING_RATE_BOUNDS = (0.01, 0.30)
OPTUNA_SUBSAMPLE_BOUNDS = (0.6, 1.0)
OPTUNA_COLSAMPLE_BYTREE_BOUNDS = (0.6, 1.0)
OPTUNA_MIN_CHILD_WEIGHT_BOUNDS = (1.0, 10.0)
OPTUNA_REG_LAMBDA_BOUNDS = (1e-3, 10.0)

XGB_BASE_PARAMS = {
    "n_estimators": 120,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "logloss",
}


def required_samples(n_windows=None):
    """Retourne le nombre total d'observations necessaires pour un protocole donne."""
    if n_windows is None:
        n_windows = N_WINDOWS
    return INITIAL_TRAIN_SIZE + int(n_windows) * MODEL_LIFE_WINDOW


def full_protocol_samples():
    """Retourne le nombre d'observations necessaires pour le protocole complet dev + holdout."""
    return required_samples(FULL_PROTOCOL_WINDOWS)


def build_xgb_params(seed, overrides=None):
    """Construit le dictionnaire final de parametres XGBoost a partir de la config."""
    params = dict(XGB_BASE_PARAMS)
    if overrides is not None:
        params.update(overrides)
    params["random_state"] = seed
    return params


def score_model_metadata():
    """Construit les metadonnees de configuration a stocker avec le score appris."""
    return {
        "random_state": RANDOM_STATE,
        "model_life_window": MODEL_LIFE_WINDOW,
        "n_windows": N_WINDOWS,
        "full_protocol_windows": FULL_PROTOCOL_WINDOWS,
        "initial_balance": INITIAL_BALANCE,
        "tp_gain": TP_GAIN,
        "fp_loss": FP_LOSS,
        "threshold": THRESHOLD,
        "corr_threshold": CORR_THRESHOLD,
        "target_corr_method": TARGET_CORR_METHOD,
        "redundancy_corr_method": REDUNDANCY_CORR_METHOD,
        "initial_train_size": INITIAL_TRAIN_SIZE,
        "score_train_windows": SCORE_TRAIN_WINDOWS,
        "score_test_windows": SCORE_TEST_WINDOWS,
        "optuna_train_windows": OPTUNA_TRAIN_WINDOWS,
        "holdout_windows": HOLDOUT_WINDOWS,
        "dataset_path": str(DATASET_PATH),
        "dataset_dev_path": str(DATASET_DEV_PATH),
        "dataset_holdout_path": str(DATASET_HOLDOUT_PATH),
        "digit_positive_min_label": DIGIT_POSITIVE_MIN_LABEL,
        "require_free_family": REQUIRE_FREE_FAMILY,
        "require_generating_subspace": REQUIRE_GENERATING_SUBSPACE,
        "orthogonalize_family": ORTHOGONALIZE_FAMILY,
        "normalize_family": NORMALIZE_FAMILY,
        "score_operator_type": SCORE_OPERATOR_TYPE,
        "score_operator_ridge_lambda": SCORE_OPERATOR_RIDGE_LAMBDA,
    }


def validate_score_model_metadata(score_model):
    """Verifie que le score sauvegarde est coherent avec la configuration actuelle."""
    metadata = score_model.get("metadata", {})
    expected = score_model_metadata()
    mismatches = []

    for key, expected_value in expected.items():
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            mismatches.append((key, actual_value, expected_value))

    if mismatches:
        formatted = ", ".join(
            f"{key}={actual!r} (attendu {expected!r})"
            for key, actual, expected in mismatches
        )
        raise ValueError(
            "Le score sauvegarde n'est pas aligne avec config.py. "
            f"Relancer main.py avec la configuration courante. Mismatchs: {formatted}"
        )


def validate_config():
    """Valide les contraintes minimales de coherence de la configuration globale."""
    if MODEL_LIFE_WINDOW <= 0:
        raise ValueError("MODEL_LIFE_WINDOW doit etre strictement positif.")

    if N_WINDOWS <= 0:
        raise ValueError("N_WINDOWS doit etre strictement positif.")

    if FULL_PROTOCOL_WINDOWS <= N_WINDOWS:
        raise ValueError(
            "FULL_PROTOCOL_WINDOWS doit etre strictement superieur a N_WINDOWS."
        )

    if SCORE_TRAIN_WINDOWS != N_WINDOWS:
        raise ValueError(
            "SCORE_TRAIN_WINDOWS doit etre egal a N_WINDOWS dans le protocole dev actuel."
        )

    if SCORE_TEST_WINDOWS != 0:
        raise ValueError(
            "SCORE_TEST_WINDOWS doit rester egal a 0 car le holdout est externalise."
        )

    if OPTUNA_TRAIN_WINDOWS != N_WINDOWS:
        raise ValueError(
            "OPTUNA_TRAIN_WINDOWS doit etre egal a N_WINDOWS dans le protocole actuel."
        )

    if not 0 <= DIGIT_POSITIVE_MIN_LABEL <= 9:
        raise ValueError("DIGIT_POSITIVE_MIN_LABEL doit appartenir a [0, 9].")

    if TARGET_CORR_METHOD not in CORRELATION_METHODS:
        raise ValueError(
            f"TARGET_CORR_METHOD doit appartenir a {CORRELATION_METHODS}."
        )

    if REDUNDANCY_CORR_METHOD not in CORRELATION_METHODS:
        raise ValueError(
            f"REDUNDANCY_CORR_METHOD doit appartenir a {CORRELATION_METHODS}."
        )

    if REQUIRE_FREE_FAMILY is not True:
        raise ValueError("REQUIRE_FREE_FAMILY doit rester fixe a True.")

    if REQUIRE_GENERATING_SUBSPACE is not True:
        raise ValueError("REQUIRE_GENERATING_SUBSPACE doit rester fixe a True.")


validate_config()
