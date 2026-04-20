from pathlib import Path


RANDOM_STATE = 42

DATA_GATHERING_DIR = Path("data_gathering")
DIGITS_SOURCE_CSV_PATH = DATA_GATHERING_DIR / "digits_source.csv"
REFERENCE_CSV_PATH = DATA_GATHERING_DIR / "reference.csv"
DATASET_PATH = DATA_GATHERING_DIR / "dataset.csv"
FEATURE_PREFIX = "pixel_"
TARGET_COLUMN = "momentum"
DIGIT_POSITIVE_MIN_LABEL = 5

INITIAL_TRAIN_SIZE = 97  # 1797 - 170 * 10 = 97
MODEL_LIFE_WINDOW = 10
N_WINDOWS = 170
SCORE_TRAIN_WINDOWS = 120
SCORE_TEST_WINDOWS = N_WINDOWS - SCORE_TRAIN_WINDOWS
OPTUNA_TRAIN_WINDOWS = SCORE_TRAIN_WINDOWS
HOLDOUT_WINDOWS = N_WINDOWS - OPTUNA_TRAIN_WINDOWS

INITIAL_BALANCE = 10_000.0
TP_GAIN = 0.04
FP_LOSS = 0.03
THRESHOLD = 0.5
CORR_THRESHOLD = 0.90
TOP_N_CORRELATED_PAIRS = 10

SCORE_MODEL_PATH = Path("artifacts/score_function.json")
BEST_PARAMS_PATH = Path("artifacts/optuna_best_xgb.json")
HOLDOUT_EVALUATION_PATH = Path("artifacts/holdout_evaluation.json")
HOLDOUT_PLOT_PATH = Path("artifacts/holdout_probability_predictions.png")

GENERATOR_DIGITS_SOURCE_PATH = DIGITS_SOURCE_CSV_PATH
GENERATOR_REFERENCE_PATH = REFERENCE_CSV_PATH
GENERATOR_OUTPUT_PATH = DATASET_PATH
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


def required_samples():
    """Retourne le nombre total d'observations necessaires pour le protocole courant."""
    return INITIAL_TRAIN_SIZE + N_WINDOWS * MODEL_LIFE_WINDOW


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
        "initial_balance": INITIAL_BALANCE,
        "tp_gain": TP_GAIN,
        "fp_loss": FP_LOSS,
        "threshold": THRESHOLD,
        "initial_train_size": INITIAL_TRAIN_SIZE,
        "score_train_windows": SCORE_TRAIN_WINDOWS,
        "score_test_windows": SCORE_TEST_WINDOWS,
        "optuna_train_windows": OPTUNA_TRAIN_WINDOWS,
        "dataset_path": str(DATASET_PATH),
        "digit_positive_min_label": DIGIT_POSITIVE_MIN_LABEL,
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

    if not 0 < SCORE_TRAIN_WINDOWS < N_WINDOWS:
        raise ValueError(
            "SCORE_TRAIN_WINDOWS doit etre strictement compris entre 0 et N_WINDOWS."
        )

    if OPTUNA_TRAIN_WINDOWS != SCORE_TRAIN_WINDOWS:
        raise ValueError(
            "OPTUNA_TRAIN_WINDOWS doit etre egal a SCORE_TRAIN_WINDOWS dans le protocole actuel."
        )

    if not 0 <= DIGIT_POSITIVE_MIN_LABEL <= 9:
        raise ValueError("DIGIT_POSITIVE_MIN_LABEL doit appartenir a [0, 9].")


validate_config()
