import json

import optuna

from backtest import evaluate_rows_with_score, run_window_backtest
import config
from metric import scoring_function


def suggest_xgb_params(trial):
    """Echantillonne un jeu d'hyperparametres XGBoost dans l'espace Optuna configure."""
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            *config.OPTUNA_N_ESTIMATORS_BOUNDS,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            *config.OPTUNA_MAX_DEPTH_BOUNDS,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            *config.OPTUNA_LEARNING_RATE_BOUNDS,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            *config.OPTUNA_SUBSAMPLE_BOUNDS,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            *config.OPTUNA_COLSAMPLE_BYTREE_BOUNDS,
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            *config.OPTUNA_MIN_CHILD_WEIGHT_BOUNDS,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            *config.OPTUNA_REG_LAMBDA_BOUNDS,
            log=True,
        ),
    }


def objective(trial, score_model):
    """Evalue un essai Optuna en backtestant XGB sur les fenetres train du score."""
    params = suggest_xgb_params(trial)
    rows = run_window_backtest(params, config.OPTUNA_TRAIN_WINDOWS)
    evaluation = evaluate_rows_with_score(rows, score_model)

    for key, value in evaluation.items():
        trial.set_user_attr(key, value)

    return evaluation["score_median"]


def print_trial_summary(study, trial):
    """Affiche un resume lisible des statistiques principales d'un essai Optuna."""
    if trial.value is None:
        return

    print(
        f"Trial {trial.number:03d} | "
        f"score_median={trial.value:.6f} | "
        f"score_corr={trial.user_attrs.get('score_corr_gain_effective', 0.0):.6f} | "
        f"tp={trial.user_attrs.get('tp', 0)} | "
        f"tn={trial.user_attrs.get('tn', 0)} | "
        f"fp={trial.user_attrs.get('fp', 0)} | "
        f"fn={trial.user_attrs.get('fn', 0)} | "
        f"gain_realized={trial.user_attrs.get('gain_realized', 0.0):.2f} | "
        f"gain_ratio={trial.user_attrs.get('gain_ratio', 0.0):.6f}"
    )


def main():
    """Lance l'optimisation Optuna puis sauvegarde le meilleur resultat observe."""
    if not config.SCORE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Score introuvable: {config.SCORE_MODEL_PATH}. Lance main.py une premiere fois."
        )

    score_model = scoring_function.load_score_model(config.SCORE_MODEL_PATH)
    config.validate_score_model_metadata(score_model)

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, score_model),
        n_trials=config.OPTUNA_N_TRIALS,
        callbacks=[print_trial_summary],
    )

    best_rows_train = run_window_backtest(study.best_params, config.OPTUNA_TRAIN_WINDOWS)
    train_evaluation = evaluate_rows_with_score(best_rows_train, score_model)
    trial_rows = []

    for trial in study.trials:
        if trial.value is None:
            continue

        trial_rows.append(
            {
                "trial_number": int(trial.number),
                "score_median": float(trial.value),
                "score_corr_gain_effective": float(
                    trial.user_attrs.get("score_corr_gain_effective", 0.0)
                ),
                "gain_ratio": float(trial.user_attrs.get("gain_ratio", 0.0)),
                "tp": int(trial.user_attrs.get("tp", 0)),
                "tn": int(trial.user_attrs.get("tn", 0)),
                "fp": int(trial.user_attrs.get("fp", 0)),
                "fn": int(trial.user_attrs.get("fn", 0)),
            }
        )

    trial_rows.sort(key=lambda row: row["trial_number"])

    result = {
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": config.OPTUNA_N_TRIALS,
        "optuna_train_windows": config.OPTUNA_TRAIN_WINDOWS,
        "holdout_windows": config.HOLDOUT_WINDOWS,
        "dataset_path": str(config.DATASET_PATH),
        "holdout_dataset_path": str(config.DATASET_HOLDOUT_PATH),
        "train_evaluation": train_evaluation,
        "trial_history": trial_rows,
    }

    config.BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.BEST_PARAMS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== Best train score median ===")
    print(result["best_value"])
    print()
    print("=== Best params ===")
    print(result["best_params"])
    print()
    print("=== Train evaluation (windows 1..120) ===")
    print(train_evaluation)
    print()
    print("=== Holdout evaluation ===")
    print("Non calcule ici : le holdout est externalise et reserve a evaluate_holdout_xgb.py.")
    print()
    print("=== Saved to ===")
    print(config.BEST_PARAMS_PATH.resolve())


if __name__ == "__main__":
    main()
