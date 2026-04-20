import json
from datetime import datetime, timezone

import pandas as pd

from backtest import evaluate_rows_with_score, run_window_backtest
import config
from data_gathering import prebuilt_dataset
from metric import scoring_function
from ploting import (
    plot_probability_predictions,
    plot_referential_score_evolution,
    plot_referential_score_evolution_html_3d,
)
from tools.name_generator import generate_unique_model_name


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


def build_holdout_run_record(
    model_name,
    best_params,
    dev_train_corr,
    train_evaluation,
    holdout_summary,
):
    """Construit la ligne CSV d'historique d'un run holdout."""
    record = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
    }

    for key, value in config.score_model_metadata().items():
        record[f"config_{key}"] = value

    for key, value in best_params.items():
        record[f"xgb_{key}"] = value

    record.update(
        {
            "gain_realized": holdout_summary["gain_realized"],
            "gain_ratio": holdout_summary["gain_ratio"],
            "score_median": holdout_summary["score_median"],
            "score_corr_gain_effective_dev_train": dev_train_corr,
            "score_corr_gain_effective_dev_optimized": (
                None
                if train_evaluation is None
                else train_evaluation.get("score_corr_gain_effective")
            ),
            "score_corr_gain_effective_holdout": holdout_summary[
                "score_corr_gain_effective"
            ],
            "tp": holdout_summary["tp"],
            "tn": holdout_summary["tn"],
            "fp": holdout_summary["fp"],
            "fn": holdout_summary["fn"],
        }
    )

    return record


def append_holdout_run_csv(record):
    """Ajoute un run holdout a l'historique CSV des modeles evalues."""
    csv_path = config.HOLDOUT_MODEL_RUNS_CSV_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_row_df = pd.DataFrame([record])

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True, sort=False)
    else:
        updated_df = new_row_df

    updated_df.to_csv(csv_path, index=False)
    return csv_path


def main():
    """Rejoue le backtest avec le meilleur XGB et produit le rapport holdout final."""
    if not config.SCORE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Score introuvable: {config.SCORE_MODEL_PATH}. Lance main.py d'abord."
        )

    score_model = scoring_function.load_score_model(config.SCORE_MODEL_PATH)
    config.validate_score_model_metadata(score_model)

    best_params, best_payload = load_best_params()
    model_name = generate_unique_model_name()
    protocol_dataset = prebuilt_dataset.load_protocol_dataset()

    rows_all, prediction_rows_all = run_window_backtest(
        best_params,
        config.FULL_PROTOCOL_WINDOWS,
        collect_predictions=True,
        dataset=protocol_dataset,
    )
    rows_holdout = rows_all[config.OPTUNA_TRAIN_WINDOWS:]
    holdout_summary = evaluate_rows_with_score(rows_holdout, score_model)
    prediction_rows_holdout = prediction_rows_all[
        config.OPTUNA_TRAIN_WINDOWS * config.MODEL_LIFE_WINDOW:
    ]

    detail_all_df = pd.DataFrame(rows_all)
    score_values_all = scoring_function.apply_score_model_frame(
        detail_all_df,
        score_model,
    )
    detail_all_df["score_metric"] = score_values_all
    all_coordinates_df = scoring_function.extract_score_coordinates_frame(
        detail_all_df,
        score_model,
    )
    full_referential_df = pd.DataFrame(
        {
            "model_life_window": detail_all_df["window_id"].values.astype(int),
            "classification": all_coordinates_df["score_classification"].values.astype(float),
            "separation": all_coordinates_df["score_separation"].values.astype(float),
            "generalization": all_coordinates_df["score_generalization"].values.astype(float),
            "stabilite": all_coordinates_df["score_stabilite"].values.astype(float),
            "score_metric": score_values_all.astype(float),
            "gain_effective": detail_all_df["gain_effective"].values.astype(float),
        }
    )

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
    holdout_coordinates_df = scoring_function.extract_score_coordinates_frame(
        detail_df,
        score_model,
    )
    holdout_referential_df = pd.DataFrame(
        {
            "model_life_window": detail_df["window_id"].values.astype(int),
            "classification": holdout_coordinates_df["score_classification"].values.astype(float),
            "separation": holdout_coordinates_df["score_separation"].values.astype(float),
            "generalization": holdout_coordinates_df["score_generalization"].values.astype(float),
            "stabilite": holdout_coordinates_df["score_stabilite"].values.astype(float),
            "score_metric": score_values_holdout.astype(float),
            "gain_effective": detail_df["gain_effective"].values.astype(float),
        }
    )

    plot_path = plot_probability_predictions(
        prediction_df["y_true"].values,
        prediction_df["y_proba"].values,
        threshold=config.THRESHOLD,
        title="Holdout XGB probability predictions",
        path=config.HOLDOUT_PLOT_PATH,
    )
    referential_plot_path = plot_referential_score_evolution(
        holdout_referential_df,
        title="Trajectoire du holdout dans R(O; classification; separation; generalization; stabilite), indexee par model-life-window",
        path=config.HOLDOUT_REFERENTIAL_PLOT_PATH,
    )
    full_protocol_referential_html_path = plot_referential_score_evolution_html_3d(
        full_referential_df,
        title=(
            "Trajectoire interactive du protocole complet "
            "dev + holdout dans le referentiel"
        ),
        path=config.FULL_PROTOCOL_REFERENTIAL_HTML_PATH,
        split_window_id=config.OPTUNA_TRAIN_WINDOWS,
    )
    dev_train_corr = score_model.get("training_target_corr")
    train_evaluation = best_payload.get("train_evaluation")
    holdout_run_record = build_holdout_run_record(
        model_name,
        best_params,
        dev_train_corr,
        train_evaluation,
        holdout_summary,
    )
    holdout_runs_csv_path = append_holdout_run_csv(holdout_run_record)

    result = {
        "model_name": model_name,
        "best_params": best_params,
        "best_value_from_optuna": best_payload.get("best_value"),
        "optuna_train_windows": config.OPTUNA_TRAIN_WINDOWS,
        "holdout_windows": config.HOLDOUT_WINDOWS,
        "dataset_dev_path": str(config.DATASET_DEV_PATH),
        "dataset_holdout_path": str(config.DATASET_HOLDOUT_PATH),
        "holdout_summary": holdout_summary,
        "holdout_details": detail_df[DETAIL_COLUMNS].to_dict(orient="records"),
        "holdout_run_record": holdout_run_record,
        "holdout_runs_csv_path": str(holdout_runs_csv_path),
        "holdout_plot_path": str(plot_path) if plot_path is not None else None,
        "holdout_referential_plot_path": (
            str(referential_plot_path) if referential_plot_path is not None else None
        ),
        "full_protocol_referential_html_path": (
            str(full_protocol_referential_html_path)
            if full_protocol_referential_html_path is not None
            else None
        ),
    }

    config.HOLDOUT_EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.HOLDOUT_EVALUATION_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("=== Model name ===")
    print(model_name)
    print()
    print("=== Best params loaded ===")
    print(best_params)
    print()
    print("=== Protocol datasets loaded ===")
    print(config.DATASET_DEV_PATH.resolve())
    print(config.DATASET_HOLDOUT_PATH.resolve())
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
    if referential_plot_path is not None:
        print("=== Holdout referential plot saved to ===")
        print(referential_plot_path.resolve())
        print()
    if full_protocol_referential_html_path is not None:
        print("=== Full protocol 3D HTML saved to ===")
        print(full_protocol_referential_html_path.resolve())
        print()
    print("=== Holdout runs CSV updated ===")
    print(holdout_runs_csv_path.resolve())
    print()
    print("=== Saved to ===")
    print(config.HOLDOUT_EVALUATION_PATH.resolve())


if __name__ == "__main__":
    main()
