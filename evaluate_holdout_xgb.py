import json

import pandas as pd

from backtest import evaluate_rows_with_score, run_window_backtest
import config
from metric import scoring_function
from ploting import plot_probability_predictions


DETAIL_COLUMNS = [
    "window_id",
    "train_size_before_update",
    "tp",
    "tn",
    "fp",
    "fn",
    "gain_realized",
    "gain_max_possible",
    "gain_effective",
    "score_metric",
]


def load_best_params():
    """Recharge les meilleurs hyperparametres XGB sauvegardes par Optuna."""
    if not config.BEST_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable: {config.BEST_PARAMS_PATH}. Lance optuna_xgb_simple.py d'abord."
        )

    payload = json.loads(config.BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    best_params = payload.get("best_params")
    if not best_params:
        raise ValueError(
            f"Le fichier {config.BEST_PARAMS_PATH} ne contient pas de 'best_params'."
        )

    return best_params, payload


def main():
    """Rejoue le backtest avec le meilleur XGB et produit le rapport holdout final."""
    if not config.SCORE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Score introuvable: {config.SCORE_MODEL_PATH}. Lance main.py d'abord."
        )

    score_model = scoring_function.load_score_model(config.SCORE_MODEL_PATH)
    config.validate_score_model_metadata(score_model)

    best_params, best_payload = load_best_params()

    rows_all, prediction_rows_all = run_window_backtest(
        best_params,
        config.N_WINDOWS,
        collect_predictions=True,
    )
    rows_holdout = rows_all[config.OPTUNA_TRAIN_WINDOWS:]
    holdout_summary = evaluate_rows_with_score(rows_holdout, score_model)
    prediction_rows_holdout = prediction_rows_all[
        config.OPTUNA_TRAIN_WINDOWS * config.MODEL_LIFE_WINDOW:
    ]

    detail_df = pd.DataFrame(rows_holdout)
    score_values_holdout, family_components = scoring_function.apply_score_model_frame(
        detail_df,
        score_model,
        return_components=True,
    )
    detail_df["score_metric"] = score_values_holdout
    for family_name, values in family_components.items():
        detail_df[family_name] = values
    prediction_df = pd.DataFrame(prediction_rows_holdout)

    plot_path = plot_probability_predictions(
        prediction_df["y_true"].values,
        prediction_df["y_proba"].values,
        threshold=config.THRESHOLD,
        title="Holdout XGB probability predictions",
        path=config.HOLDOUT_PLOT_PATH,
    )

    result = {
        "best_params": best_params,
        "best_value_from_optuna": best_payload.get("best_value"),
        "optuna_train_windows": config.OPTUNA_TRAIN_WINDOWS,
        "holdout_windows": config.HOLDOUT_WINDOWS,
        "holdout_summary": holdout_summary,
        "holdout_details": detail_df[DETAIL_COLUMNS].to_dict(orient="records"),
        "holdout_plot_path": str(plot_path) if plot_path is not None else None,
    }

    config.HOLDOUT_EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.HOLDOUT_EVALUATION_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("=== Best params loaded ===")
    print(best_params)
    print()
    print("=== Holdout summary (windows 121..170) ===")
    print(holdout_summary)
    print()
    print("=== Holdout details ===")
    print(detail_df[DETAIL_COLUMNS].to_string(index=False))
    print()
    if plot_path is not None:
        print("=== Holdout plot saved to ===")
        print(plot_path.resolve())
        print()
    print("=== Saved to ===")
    print(config.HOLDOUT_EVALUATION_PATH.resolve())


if __name__ == "__main__":
    main()
