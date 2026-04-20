import json
from pathlib import Path

import numpy as np
import optuna

from sklearn.datasets import load_digits
from xgboost import XGBClassifier

from metric import scoring_function
from metric import window_metrics


RANDOM_STATE = 42
N_TRIALS = 20
SCORE_MODEL_PATH = Path("artifacts/score_function.json")
BEST_PARAMS_PATH = Path("artifacts/optuna_best_xgb.json")


def load_config_from_score_model(score_model):
    metadata = score_model.get("metadata", {})
    return {
        "model_life_window": int(metadata.get("model_life_window", 10)),
        "n_windows": int(metadata.get("n_windows", 170)),
        "initial_balance": float(metadata.get("initial_balance", 10_000.0)),
        "tp_gain": float(metadata.get("tp_gain", 0.04)),
        "fp_loss": float(metadata.get("fp_loss", 0.03)),
        "threshold": float(metadata.get("threshold", 0.5)),
        "initial_train_size": int(metadata.get("initial_train_size", 97)),
    }


def train_model(X_train, y_train, params, seed):
    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        reg_lambda=params["reg_lambda"],
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def summarize_rows(rows):
    gain_realized = float(sum(row["gain_realized"] for row in rows))
    gain_max_possible = float(sum(row["gain_max_possible"] for row in rows))

    if abs(gain_max_possible) < 1e-12:
        gain_ratio = 0.0
    else:
        gain_ratio = gain_realized / gain_max_possible

    return {
        "tp": int(sum(row["tp"] for row in rows)),
        "tn": int(sum(row["tn"] for row in rows)),
        "fp": int(sum(row["fp"] for row in rows)),
        "fn": int(sum(row["fn"] for row in rows)),
        "gain_realized": gain_realized,
        "gain_max_possible": gain_max_possible,
        "gain_ratio": float(gain_ratio),
    }


def evaluate_xgb_params(params, score_model, config):
    digits = load_digits()
    X = digits.data
    y = (digits.target >= 5).astype(int)

    rng = np.random.default_rng(RANDOM_STATE)
    all_indices = np.arange(len(y))
    rng.shuffle(all_indices)

    initial_train_size = config["initial_train_size"]
    model_life_window = config["model_life_window"]
    n_windows = config["n_windows"]
    required = initial_train_size + n_windows * model_life_window

    if len(y) < required:
        raise ValueError(
            f"Dataset trop petit : {len(y)} lignes, il en faut au moins {required}."
        )

    train_indices = all_indices[:initial_train_size].tolist()
    pool_indices = all_indices[initial_train_size:].tolist()

    X_train = X[train_indices]
    y_train = y[train_indices]

    balance = config["initial_balance"]
    rows = []

    for window_id in range(1, n_windows + 1):
        chosen_pos = rng.choice(len(pool_indices), size=model_life_window, replace=False)
        chosen_pos_set = set(chosen_pos.tolist())

        window_indices = [pool_indices[pos] for pos in chosen_pos]
        pool_indices = [idx for j, idx in enumerate(pool_indices) if j not in chosen_pos_set]

        X_window = X[window_indices]
        y_window = y[window_indices]

        model = train_model(X_train, y_train, params, seed=RANDOM_STATE + window_id)
        y_proba = model.predict_proba(X_window)[:, 1]

        metric_row = window_metrics.build_window_metric_dict(
            y_window,
            y_proba,
            balance_start=balance,
            threshold=config["threshold"],
            tp_gain=config["tp_gain"],
            fp_loss=config["fp_loss"],
        )
        balance = metric_row["balance_end"]
        rows.append(metric_row)

        X_train = np.vstack([X_train, X_window])
        y_train = np.concatenate([y_train, y_window])

    X_score = np.asarray(
        [[row[name] for name in score_model["feature_names"]] for row in rows],
        dtype=float,
    )
    score_values = scoring_function.apply_linear_score_model(X_score, score_model)
    return float(np.median(score_values)), summarize_rows(rows)


def objective(trial, score_model, config):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 250),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    score_value, summary = evaluate_xgb_params(params, score_model, config)

    for key, value in summary.items():
        trial.set_user_attr(key, value)

    return score_value


def print_trial_summary(study, trial):
    if trial.value is None:
        return

    print(
        f"Trial {trial.number:03d} | "
        f"score={trial.value:.6f} | "
        f"tp={trial.user_attrs.get('tp', 0)} | "
        f"tn={trial.user_attrs.get('tn', 0)} | "
        f"fp={trial.user_attrs.get('fp', 0)} | "
        f"fn={trial.user_attrs.get('fn', 0)} | "
        f"gain_realized={trial.user_attrs.get('gain_realized', 0.0):.2f} | "
        f"gain_max_possible={trial.user_attrs.get('gain_max_possible', 0.0):.2f} | "
        f"gain_ratio={trial.user_attrs.get('gain_ratio', 0.0):.6f}"
    )


def main():
    if not SCORE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Score introuvable: {SCORE_MODEL_PATH}. Lance main.py une premiere fois."
        )

    score_model = scoring_function.load_score_model(SCORE_MODEL_PATH)
    config = load_config_from_score_model(score_model)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, score_model, config),
        n_trials=N_TRIALS,
        callbacks=[print_trial_summary],
    )

    result = {
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": N_TRIALS,
    }

    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== Best score ===")
    print(result["best_value"])
    print()
    print("=== Best params ===")
    print(result["best_params"])
    print()
    print("=== Saved to ===")
    print(BEST_PARAMS_PATH.resolve())


if __name__ == "__main__":
    main()
