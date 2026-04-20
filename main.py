import numpy as np
import pandas as pd

from xgboost import XGBClassifier

import config
from data_gathering import prebuilt_dataset
from metric import algebra
from metric import correlation_analysis
from metric import scoring_function
from metric import window_metrics
from ploting import plot_referential_score_evolution


# ============================================================
# 2. Chargement dataset preconstruit
# ============================================================

temporal_data = prebuilt_dataset.load_prebuilt_dataset(config.DATASET_DEV_PATH)

X = temporal_data["data"]
y = temporal_data["momentum"]
temporal_stats = temporal_data["sequence_stats"]

n_samples = len(y)
required = config.required_samples()

if n_samples < required:
    raise ValueError(
        f"Dataset trop petit : {n_samples} lignes, "
        f"il en faut au moins {required}."
    )

print(f"Nombre total de lignes : {n_samples}")
print(f"Mode temporel         : {temporal_data['mode']}")
print(f"Source dataset        : {temporal_data['path'].resolve()}")
print(f"Taux momentum         : {temporal_stats['momentum_rate']:.4f}")
print(f"Switches temporels    : {temporal_stats['switches']}")
print(f"Train initial         : {config.INITIAL_TRAIN_SIZE}")
print(f"Fenêtres dev          : {config.N_WINDOWS}")
print(f"Fenêtres holdout      : {config.HOLDOUT_WINDOWS}")
print(f"Taille fenêtre        : {config.MODEL_LIFE_WINDOW}")
print()

target_corr_method_label = config.TARGET_CORR_METHOD.capitalize()
redundancy_corr_method_label = config.REDUNDANCY_CORR_METHOD.capitalize()


# ============================================================
# 3. Modele
# ============================================================

def train_model(X_train, y_train, seed=42):
    """Entraine un XGBoost sur le train courant avec la configuration par defaut."""
    model = XGBClassifier(**config.build_xgb_params(seed))
    model.fit(X_train, y_train)
    return model


def compute_basis_geometry_metrics(matrix, ridge_lambda):
    """Calcule les invariants geometriques principaux d'une base de travail."""
    matrix = np.asarray(matrix, dtype=float)
    gram = algebra.gram_matrix(matrix)
    covariance = algebra.covariance_matrix(matrix)

    return {
        "gram": gram,
        "covariance": covariance,
        "det_gram": algebra.matrix_determinant(gram),
        "det_cov": algebra.matrix_determinant(covariance),
        "logdet_cov_plus_lambda_i": algebra.regularized_logdet(
            covariance,
            ridge_lambda=ridge_lambda,
        ),
    }


def build_referential_frame(
    window_ids,
    basis_coordinates,
    basis_feature_names,
    score_values,
    gain_effective_values,
):
    """Construit les coordonnees du point dans la base finale avant application de l'operateur."""
    basis_coordinates = np.asarray(basis_coordinates, dtype=float)

    if basis_coordinates.ndim != 2:
        raise ValueError("basis_coordinates doit etre une matrice 2D.")

    clean_feature_names = [
        name.replace("score_", "")
        for name in basis_feature_names
    ]

    referential_payload = {
        "model_life_window": np.asarray(window_ids, dtype=int),
    }

    for column_index, column_name in enumerate(clean_feature_names):
        referential_payload[column_name] = basis_coordinates[:, column_index]

    referential_payload["score_metric"] = np.asarray(score_values, dtype=float)
    referential_payload["gain_effective"] = np.asarray(gain_effective_values, dtype=float)

    return pd.DataFrame(referential_payload)


def format_linear_operator_phi(
    feature_names,
    weights,
    symbol="phi",
    point_symbol="z_t",
    coordinate_suffix="_t",
):
    """Formate l'operateur lineaire du score sur une fenetre donnee."""
    clean_feature_names = [
        name.replace("score_", "")
        for name in feature_names
    ]
    terms = [
        f"{weight:+.6f}·{name}{coordinate_suffix}"
        for name, weight in zip(clean_feature_names, weights)
    ]
    rhs = " ".join(terms) if terms else "0"
    return f"{symbol}({point_symbol}) = {rhs}"


def build_empirical_basis(
    X_values,
    column_names,
    target_values,
    corr_threshold,
    target_corr_method,
    redundancy_corr_method,
):
    """Construit une base libre par filtrage de corrélation puis ajout glouton par rang."""
    X_values = np.asarray(X_values, dtype=float)
    target_values = np.asarray(target_values, dtype=float)
    X_clean = np.nan_to_num(X_values, nan=0.0)
    name_to_index = {name: j for j, name in enumerate(column_names)}

    non_constant_names = [
        name
        for j, name in enumerate(column_names)
        if np.std(X_clean[:, j]) > 1e-12
    ]

    if non_constant_names:
        target_corr = {
            name: scoring_function.correlation_safe(
                X_values[:, name_to_index[name]],
                target_values,
                method=target_corr_method,
            )
            for name in non_constant_names
        }
        non_constant_indices = [name_to_index[name] for name in non_constant_names]
        corr_df = correlation_analysis.correlation_matrix_ignore_nan(
            X_values[:, non_constant_indices],
            non_constant_names,
            method=redundancy_corr_method,
        )
        filtered_names, dropped_corr = correlation_analysis.select_uncorrelated_columns_by_target_corr(
            corr_df,
            target_corr,
            threshold=corr_threshold,
        )
    else:
        target_corr = {}
        filtered_names = []
        dropped_corr = []

    selected_names = []
    current = None

    for name in filtered_names:
        col = X_clean[:, [name_to_index[name]]]

        if current is None:
            selected_names.append(name)
            current = col
            continue

        rank_before = np.linalg.matrix_rank(current)
        candidate = np.column_stack([current, col])
        rank_after = np.linalg.matrix_rank(candidate)

        if rank_after > rank_before:
            selected_names.append(name)
            current = candidate

    return {
        "X_clean": X_clean,
        "target_corr": target_corr,
        "non_constant_names": non_constant_names,
        "filtered_names": filtered_names,
        "dropped_corr": dropped_corr,
        "selected_names": selected_names,
        "current": current,
    }


def build_family_score_bundle(
    df_train,
    df_test,
    df_all,
    family_name,
    family_cols,
    target_train,
    corr_threshold,
    target_corr_method,
    redundancy_corr_method,
):
    """Apprend le sous-score d'une famille et l'applique aux splits train, test et global."""
    family_basis = build_empirical_basis(
        df_train[family_cols].values,
        family_cols,
        target_train,
        corr_threshold=corr_threshold,
        target_corr_method=target_corr_method,
        redundancy_corr_method=redundancy_corr_method,
    )

    selected_cols = family_basis["selected_names"]
    if not selected_cols:
        raise ValueError(
            f"Aucune metrique exploitable pour la famille '{family_name}'."
        )

    family_model, train_scores = scoring_function.fit_linear_score_model(
        df_train[selected_cols].values,
        selected_cols,
        target_train,
        target_name=f"gain_effective_{family_name}",
        operator_type=config.SCORE_OPERATOR_TYPE,
        ridge_lambda=config.SCORE_OPERATOR_RIDGE_LAMBDA,
        target_corr_method=target_corr_method,
    )
    test_scores = scoring_function.apply_linear_score_model(
        df_test[family_model["feature_names"]].values,
        family_model,
    )
    all_scores = scoring_function.apply_linear_score_model(
        df_all[family_model["feature_names"]].values,
        family_model,
    )

    score_column = f"score_{family_name}"

    return {
        "family_name": family_name,
        "score_column": score_column,
        "basis": family_basis,
        "model": family_model,
        "train_scores": train_scores,
        "test_scores": test_scores,
        "all_scores": all_scores,
    }


# ============================================================
# 4. Split initial séquentiel
# ============================================================

X_train = X[:config.INITIAL_TRAIN_SIZE]
y_train = y[:config.INITIAL_TRAIN_SIZE]
remaining_pool_size = n_samples - config.INITIAL_TRAIN_SIZE

print(f"Taille train initial : {len(X_train)}")
print(f"Taille pool restante : {remaining_pool_size}")
print()


# ============================================================
# 5. Boucle séquentielle par fenêtres
# ============================================================

balance = config.INITIAL_BALANCE
rows = []

for window_id in range(1, config.N_WINDOWS + 1):
    window_start = config.INITIAL_TRAIN_SIZE + (window_id - 1) * config.MODEL_LIFE_WINDOW
    window_end = window_start + config.MODEL_LIFE_WINDOW

    X_window = X[window_start:window_end]
    y_window = y[window_start:window_end]

    model = train_model(X_train, y_train, seed=config.RANDOM_STATE + window_id)

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_proba = model.predict_proba(X_window)[:, 1]

    metric_row = window_metrics.build_window_metric_dict(
        y_window,
        y_proba,
        balance_start=balance,
        threshold=config.THRESHOLD,
        tp_gain=config.TP_GAIN,
        fp_loss=config.FP_LOSS,
        history_rows=rows,
        y_train_true=y_train,
        y_train_proba=y_train_proba,
    )
    balance = metric_row["balance_end"]

    row = {
        "window_id": window_id,
        "train_size_before_update": len(X_train),
        "test_window_size": config.MODEL_LIFE_WINDOW,
    }
    row.update(metric_row)
    rows.append(row)

    X_train = np.vstack([X_train, X_window])
    y_train = np.concatenate([y_train, y_window])

df_windows = pd.DataFrame(rows)


# ============================================================
# 6. Vecteurs par famille
# ============================================================

classification_cols = window_metrics.CLASSIFICATION_COLS
separation_cols = window_metrics.SEPARATION_COLS
generalization_cols = window_metrics.GENERALIZATION_COLS
stability_cols = window_metrics.STABILITY_COLS
gain_cols = window_metrics.GAIN_COLS

M_classification = df_windows[classification_cols].values
M_separation = df_windows[separation_cols].values
M_generalization = df_windows[generalization_cols].values
M_stability = df_windows[stability_cols].values
M_gain = df_windows[gain_cols].values

print("=== Formes des vecteurs par famille ===")
print("classification      :", M_classification.shape)
print("separation          :", M_separation.shape)
print("generalization      :", M_generalization.shape)
print("stabilite           :", M_stability.shape)
print("gain                :", M_gain.shape)
print()


# ============================================================
# 7. Matrice globale des métriques
# ============================================================

all_metric_cols = window_metrics.ALL_METRIC_COLS
base_metric_cols = window_metrics.REALIZED_WINDOW_METRIC_COLS

X_metrics = df_windows[base_metric_cols].values

print("=== Aperçu des fenêtres ===")
print(df_windows.head())
print()

print("=== Matrice globale des métriques réalisées de fenêtre ===")
print(df_windows[base_metric_cols].head())
print()


# ============================================================
# 8. Vérification empirique
# ============================================================

X_metrics_clean = np.nan_to_num(X_metrics, nan=0.0)

rank = np.linalg.matrix_rank(X_metrics_clean)
n_cols = X_metrics_clean.shape[1]

print(f"Rang de la matrice X_metrics : {rank}")
print(f"Nombre de métriques          : {n_cols}")

if rank == n_cols:
    print("=> Les métriques sont empiriquement libres sur ces fenêtres.")
else:
    print("=> Il existe de la redondance empirique entre les métriques.")
print()


# ============================================================
# 9. Colonnes constantes / quasi constantes
# ============================================================

print("=== Colonnes constantes / quasi constantes ===")
constant_cols = []
quasi_constant_cols = []

for j, name in enumerate(base_metric_cols):
    s = np.std(X_metrics_clean[:, j])
    if s < 1e-12:
        constant_cols.append(name)
        print(f"{name:25s} -> constante")
    elif s < 1e-6:
        quasi_constant_cols.append(name)
        print(f"{name:25s} -> quasi constante")

if not constant_cols and not quasi_constant_cols:
    print("Aucune")
print()


# ============================================================
# 10. Base empirique gloutonne
# ============================================================

basis_all = build_empirical_basis(
    X_metrics,
    base_metric_cols,
    df_windows["gain_effective"].values.astype(float),
    corr_threshold=config.CORR_THRESHOLD,
    target_corr_method=config.TARGET_CORR_METHOD,
    redundancy_corr_method=config.REDUNDANCY_CORR_METHOD,
)
selected_metric_names = basis_all["selected_names"]
current = basis_all["current"]

print(
    f"=== Filtrage automatique des colonnes trop corrélées "
    f"selon {redundancy_corr_method_label} (>=|{config.CORR_THRESHOLD}|) ==="
)
if basis_all["dropped_corr"]:
    for item in basis_all["dropped_corr"]:
        print(
            f"{item['dropped']:25s} -> retiree, trop correlee a "
            f"{item['kept']:25s} "
            f"(corr={item['corr']:+.6f}, abs={item['abs_corr']:.6f}, "
            f"target_kept={item['kept_target_corr']:.6f}, "
            f"target_drop={item['dropped_target_corr']:.6f})"
        )
else:
    print("Aucune")
print()

print("=== Base empirique gloutonne sur métriques réalisées ===")
print("Métriques candidates après filtrage :", basis_all["filtered_names"])
print("Métriques retenues                  :", selected_metric_names)
print()

if current is not None:
    print("=== Matrice associée à la base empirique ===")
    print(pd.DataFrame(current, columns=selected_metric_names).head())
    print()
else:
    print("Aucune base extraite.")
    print()


# ============================================================
# 11. Résumé final
# ============================================================

print("=== Résumé final ===")
print(f"Balance finale           : {df_windows['balance_end'].iloc[-1]:.2f}")
print(f"Gain effectif moyen      : {df_windows['gain_effective'].mean():.6f}")
print(f"Gain réalisé total       : {df_windows['gain_realized'].sum():.2f}")
print(f"Nombre moyen de trades   : {df_windows['n_predicted_positive'].mean():.4f}")
print(f"Momentum moyen par fen.  : {df_windows['n_true_momentum'].mean():.4f}")
print()


# ============================================================
# 12. Corrélations avec gain_effective
# ============================================================

print(f"=== Corrélations {target_corr_method_label} avec gain_effective ===")
corr_with_gain = correlation_analysis.correlations_to_target(
    df_windows[all_metric_cols],
    feature_names=all_metric_cols,
    target_name="gain_effective",
    method=config.TARGET_CORR_METHOD,
).sort_values(ascending=False)
print(corr_with_gain)
print()


# ============================================================
# 13. Matrice de corrélation complète
# ============================================================

corr_df = correlation_analysis.correlation_matrix_ignore_nan(
    X_metrics,
    base_metric_cols,
    method=config.REDUNDANCY_CORR_METHOD,
)
print(f"=== Corrélation {redundancy_corr_method_label} entre métriques réalisées de fenêtre ===")
print(corr_df.round(3))
print()


# ============================================================
# 14. Paires fortement corrélées
# ============================================================

print(
    f"=== Paires de métriques réalisées de fenêtre fortement corrélées "
    f"(>=|{config.CORR_THRESHOLD}|) ==="
)
correlated_pairs = correlation_analysis.extract_correlated_pairs(
    corr_df,
    threshold=config.CORR_THRESHOLD,
)
threshold_pairs = [
    pair for pair in correlated_pairs if pair["is_above_threshold"]
]

if threshold_pairs:
    for pair in threshold_pairs:
        print(
            f"{pair['left']:25s} <-> {pair['right']:25s} : "
            f"corr={pair['corr']:+.6f} abs={pair['abs_corr']:.6f}"
        )
else:
    print("Aucune paire fortement corrélée trouvée.")
print()


# ============================================================
# 15. Split score train / test
# ============================================================

df_score_train = df_windows.copy()
df_score_test = df_windows.iloc[0:0].copy()
target_gain_train = df_score_train["gain_effective"].values.astype(float)
target_gain_test = np.asarray([], dtype=float)

print("=== Jeu de développement du score ===")
print(f"Fenêtres disponibles dans dataset_dev : {len(df_score_train)}")
print(
    "Validation interne jamais vue        : "
    "aucune, le holdout est externalisé dans dataset_holdout.csv"
)
print(
    f"Windows ids du dataset_dev           : "
    f"{df_score_train['window_id'].min()} -> {df_score_train['window_id'].max()}"
)
print()


# ============================================================
# 16. Construction de la base par famille
# ============================================================

family_column_map = {
    "classification": window_metrics.CLASSIFICATION_COLS,
    "separation": window_metrics.SEPARATION_COLS,
    "generalization": window_metrics.GENERALIZATION_COLS,
    "stabilite": window_metrics.STABILITY_COLS,
}

family_bundles = {}
family_score_cols = []

for family_name, family_cols in family_column_map.items():
    bundle = build_family_score_bundle(
        df_score_train,
        df_score_test,
        df_windows,
        family_name,
        family_cols,
        target_gain_train,
        corr_threshold=config.CORR_THRESHOLD,
        target_corr_method=config.TARGET_CORR_METHOD,
        redundancy_corr_method=config.REDUNDANCY_CORR_METHOD,
    )
    family_bundles[family_name] = bundle
    family_score_cols.append(bundle["score_column"])

    print(f"=== Famille {family_name} ===")
    print("Colonnes candidates :", family_cols)
    print("Colonnes non constantes :", bundle["basis"]["non_constant_names"])
    print(f"Corrélation {target_corr_method_label} à gain_effective :")
    for name in sorted(
        bundle["basis"]["target_corr"],
        key=lambda item: -abs(bundle["basis"]["target_corr"][item]),
    ):
        print(
            f"  {name:25s} -> "
            f"{bundle['basis']['target_corr'][name]:+.6f}"
        )
    print(
        f"Filtrage corrélé automatique (>=|{config.CORR_THRESHOLD}|, "
        f"priorité à la corrélation cible {target_corr_method_label}) :"
    )
    if bundle["basis"]["dropped_corr"]:
        for item in bundle["basis"]["dropped_corr"]:
            print(
                f"  {item['dropped']:25s} -> retiree, trop correlee a "
                f"{item['kept']:25s} "
                f"(corr={item['corr']:+.6f}, "
                f"target_kept={item['kept_target_corr']:.6f}, "
                f"target_drop={item['dropped_target_corr']:.6f})"
            )
    else:
        print("  Aucune")
    print("Colonnes retenues pour le sous-score :", bundle["model"]["feature_names"])
    print()


# ============================================================
# 17. Apprentissage du score final sur les sous-scores de famille
# ============================================================

family_score_train_df = pd.DataFrame(
    {
        bundle["score_column"]: bundle["train_scores"]
        for bundle in family_bundles.values()
    }
)
family_score_test_df = pd.DataFrame(
    {
        bundle["score_column"]: bundle["test_scores"]
        for bundle in family_bundles.values()
    }
)
family_score_all_df = pd.DataFrame(
    {
        bundle["score_column"]: bundle["all_scores"]
        for bundle in family_bundles.values()
    }
)

final_score_model, final_train_scores = scoring_function.fit_linear_score_model(
    family_score_train_df[family_score_cols].values,
    family_score_cols,
    target_gain_train,
    target_name="gain_effective",
    operator_type=config.SCORE_OPERATOR_TYPE,
    ridge_lambda=config.SCORE_OPERATOR_RIDGE_LAMBDA,
    orthogonalize=config.ORTHOGONALIZE_FAMILY,
    normalize=config.NORMALIZE_FAMILY,
    target_corr_method=config.TARGET_CORR_METHOD,
)
final_family_score_cols_kept = final_score_model["feature_names"]
Z_basis_train = scoring_function.score_coordinates_with_model(
    family_score_train_df[final_family_score_cols_kept].values,
    final_score_model,
)
Z_basis_all = scoring_function.score_coordinates_with_model(
    family_score_all_df[final_family_score_cols_kept].values,
    final_score_model,
)
final_basis_geometry = compute_basis_geometry_metrics(
    Z_basis_train,
    ridge_lambda=config.GEOMETRY_RIDGE_LAMBDA,
)
final_family_rank = algebra.family_rank(Z_basis_train)
final_family_is_free = algebra.is_free_family(Z_basis_train)
final_family_is_generating_ambient = algebra.is_generating_family(
    Z_basis_train,
    ambient_dimension=Z_basis_train.shape[0],
)
final_family_is_generating_subspace = algebra.is_generating_family(
    Z_basis_train,
    ambient_dimension=final_family_rank,
)
final_family_orthogonality = algebra.orthogonality_analysis(Z_basis_train)
final_family_unit_norm = algebra.unit_norm_analysis(Z_basis_train)

if config.REQUIRE_FREE_FAMILY and not final_family_is_free:
    raise ValueError("La base finale doit etre une famille libre.")

if config.REQUIRE_GENERATING_SUBSPACE and not final_family_is_generating_subspace:
    raise ValueError("La base finale doit engendrer son sous-espace.")

if config.ORTHOGONALIZE_FAMILY and not final_family_orthogonality["is_orthogonal"]:
    raise ValueError(
        "ORTHOGONALIZE_FAMILY=True mais la base finale n'est pas orthogonale."
    )

score_values_test = scoring_function.apply_linear_score_model(
    family_score_test_df[final_family_score_cols_kept].values,
    final_score_model,
)
score_values_all = scoring_function.apply_linear_score_model(
    family_score_all_df[final_family_score_cols_kept].values,
    final_score_model,
)
w = np.asarray(final_score_model["weights"], dtype=float)
score_gain_corr_train = float(final_score_model["training_target_corr"])
score_gain_corr_test = scoring_function.correlation_safe(
    score_values_test,
    target_gain_test,
    method=config.TARGET_CORR_METHOD,
)
score_gain_corr_all = scoring_function.correlation_safe(
    score_values_all,
    df_windows["gain_effective"].values.astype(float),
    method=config.TARGET_CORR_METHOD,
)

score_model = {
    "model_type": "hierarchical_family_score",
    "family_models": {
        bundle["score_column"]: bundle["model"]
        for bundle in family_bundles.values()
    },
    "family_bases": {
        bundle["score_column"]: {
            "candidate_feature_names": family_column_map[bundle["family_name"]],
            "non_constant_feature_names": bundle["basis"]["non_constant_names"],
            "target_corr": bundle["basis"]["target_corr"],
            "filtered_feature_names": bundle["basis"]["filtered_names"],
            "selected_feature_names": bundle["model"]["feature_names"],
            "dropped_corr": bundle["basis"]["dropped_corr"],
        }
        for bundle in family_bundles.values()
    },
    "final_model": final_score_model,
    "training_target_corr": score_gain_corr_train,
    "final_basis_geometry": {
        "det_gram": final_basis_geometry["det_gram"],
        "det_cov": final_basis_geometry["det_cov"],
        "logdet_cov_plus_lambda_i": final_basis_geometry["logdet_cov_plus_lambda_i"],
        "lambda": config.GEOMETRY_RIDGE_LAMBDA,
    },
    "study_referential": {
        "name": "R(O; classification; separation; generalization; stabilite)",
        "origin": {
            "classification": 0.0,
            "separation": 0.0,
            "generalization": 0.0,
            "stabilite": 0.0,
        },
        "axes": [
            "classification",
            "separation",
            "generalization",
            "stabilite",
        ],
        "trajectory_parameter": "model_life_window",
        "orthogonalized": bool(final_score_model.get("orthogonalized", False)),
        "orthogonalization_method": final_score_model.get("orthogonalization_method"),
        "normalized": bool(final_score_model.get("normalized", False)),
    },
}

saved_score_path = scoring_function.save_score_model(
    score_model,
    config.SCORE_MODEL_PATH,
    metadata=config.score_model_metadata(),
)

print("=== Sous-scores de famille retenus pour le score final ===")
print(family_score_cols)
print()

print("=== Poids du score final appris sur la base des 4 familles ===")
for name, weight in zip(final_family_score_cols_kept, w):
    print(f"{name:25s} : {weight:+.6f}")
print()

print("=== Operateur lineaire du score ===")
print(f"Type                    : {final_score_model.get('operator_type', 'linear')}")
if final_score_model.get("operator_type") == "ridge":
    print(f"Ridge lambda            : {final_score_model.get('ridge_lambda', 0.0)}")
print(f"Base orthogonalisee     : {final_score_model.get('orthogonalized', False)}")
if final_score_model.get("orthogonalized", False):
    print(
        f"Methode                 : "
        f"{final_score_model.get('orthogonalization_method', 'inconnue')}"
    )
print(f"Base normalisee         : {final_score_model.get('normalized', False)}")
print()

print("=== Operateur phi dans la base finale transformee ===")
phi_basis_names = [
    name.replace("score_", "")
    for name in final_family_score_cols_kept
]
phi_dimension = len(phi_basis_names)
phi_point_symbol = "z_t"
phi_coordinates = ", ".join(f"{name}_t" for name in phi_basis_names)
print(f"phi : R^{phi_dimension} -> R")
print(f"{phi_point_symbol} = ({phi_coordinates})")
print(f"phi({phi_point_symbol}) = w^T {phi_point_symbol}")
print(format_linear_operator_phi(final_family_score_cols_kept, w, symbol="phi"))
print("Sur toutes les fenetres a la fois : score = Z_basis @ w")
print()

print("=== Base finale après centrage-réduction ===")
print("Sous-scores gardés :", final_family_score_cols_kept)
print("Forme de Z_basis_train :", Z_basis_train.shape)
print()

print("=== Géométrie de la base finale ===")
print(f"det(Gram)               : {final_basis_geometry['det_gram']}")
print(f"det(Cov)                : {final_basis_geometry['det_cov']}")
print(
    f"logdet(Cov + λI)         : "
    f"{final_basis_geometry['logdet_cov_plus_lambda_i']}"
)
print(f"λ utilisé               : {config.GEOMETRY_RIDGE_LAMBDA}")
print()

print("=== Etude de la famille ===")
print(f"Nombre de vecteurs      : {Z_basis_train.shape[1]}")
print(f"Dimension ambiante      : {Z_basis_train.shape[0]}")
print(f"Rang de la famille      : {final_family_rank}")
print(f"Famille libre           : {final_family_is_free}")
print(f"Generatrice de R^n      : {final_family_is_generating_ambient}")
print(f"Generatrice de son sous-espace : {final_family_is_generating_subspace}")
print(f"Famille orthogonale     : {final_family_orthogonality['is_orthogonal']}")
print(
    f"Max |cosinus hors diag| : "
    f"{final_family_orthogonality['max_abs_cosine_off_diagonal']:.6f}"
)
print(f"Tolerance orthogonalite : {final_family_orthogonality['tolerance']}")
print(f"Famille normalisee      : {final_family_unit_norm['is_unit_norm_family']}")
print(
    f"Normes des vecteurs     : "
    f"{np.array2string(final_family_unit_norm['norms'], precision=6)}"
)
print(
    f"Max |norme - 1|         : "
    f"{final_family_unit_norm['max_abs_norm_deviation']:.6f}"
)
print(f"Tolerance normalisation : {final_family_unit_norm['tolerance']}")
print()

print(
    f"=== Corrélation {target_corr_method_label}(score, gain_effective) "
    "sur le dataset de développement ==="
)
print(score_gain_corr_train)
print()

print(
    f"=== Corrélation {target_corr_method_label}(score, gain_effective) "
    "sur l'ensemble du dataset de développement ==="
)
print(score_gain_corr_all)
print()

print("=== Score sauvegardé ===")
print(saved_score_path.resolve())
print()


# ============================================================
# 18. Ajouter le score au DataFrame
# ============================================================

for column in family_score_all_df.columns:
    df_windows[column] = family_score_all_df[column].values

df_windows["score_metric"] = score_values_all
referential_df = build_referential_frame(
    df_windows["window_id"].values,
    Z_basis_all,
    final_family_score_cols_kept,
    score_values_all,
    df_windows["gain_effective"].values,
)
referential_plot_path = plot_referential_score_evolution(
    referential_df,
    title="Evolution des coordonnees Z du modele dans R(O; classification; separation; generalization; stabilite), indexee par model-life-window",
    path=config.REFERENTIAL_PLOT_PATH,
    split_window_id=None,
)

print("=== Référentiel d'étude ===")
print("R(O; classification; separation; generalization; stabilite)")
print("Origine O :", "(0, 0, 0, 0)")
print("Axes : classification, separation, generalization, stabilite")
print("Parametre de trajectoire : model_life_window")
if final_score_model.get("orthogonalized", False):
    print("Base finale orthogonalisee : oui, par decomposition QR.")
else:
    print("Base finale orthogonalisee : non.")

if final_score_model.get("normalized", False):
    print("Base finale normalisee : oui, les vecteurs du referentiel ont une norme unitaire sur le train.")
else:
    print("Base finale normalisee : non.")

print("Les coordonnees affichees ci-dessous sont les coordonnees Z dans la base finale transformee, avant application de l'operateur lineaire du score.")
print("Aperçu des coordonnées dans le référentiel :")
print(referential_df.head())
print()

print("=== Aperçu score_metric vs gain_effective ===")
print(df_windows[["window_id", "score_metric", "gain_effective"]].head())
print()
print("=== Holdout externe ===")
print(
    "Le holdout n'est plus visible dans main.py. "
    "Utiliser evaluate_holdout_xgb.py pour juger dataset_holdout.csv."
)
print()

print("=== Trajectoire du score dans le référentiel ===")
print(referential_plot_path.resolve())
print()


# ============================================================
# 19. Corrélations du score avec toutes les métriques
# ============================================================

print(f"=== Corrélations {target_corr_method_label} avec score_metric ===")
metric_plus_score_cols = all_metric_cols + ["score_metric"]
corr_with_score = correlation_analysis.correlations_to_target(
    df_windows[metric_plus_score_cols],
    feature_names=metric_plus_score_cols,
    target_name="score_metric",
    method=config.TARGET_CORR_METHOD,
).sort_values(ascending=False)
print(corr_with_score)
print()


# ============================================================
# 20. Affichage des vecteurs constituant la base
# ============================================================

'''
print("=== Vecteurs de la base empirique (colonnes) ===")

if current is not None:
    for idx, (name, col) in enumerate(zip(selected_metric_names, current.T)):
        print(f"\nVecteur {idx+1} : {name}")
        print(col)
else:
    print("Aucune base disponible.")

print("\n=== Vecteurs de base après centrage-réduction ===")

if Z_basis_train is not None:
    for idx, (name, col) in enumerate(zip(final_family_score_cols_kept, Z_basis_train.T)):
        print(f"\nVecteur normalisé {idx+1} : {name}")
        print(col)
else:
    print("Aucune base normalisée disponible.")
'''
