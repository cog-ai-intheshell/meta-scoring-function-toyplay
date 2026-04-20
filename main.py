from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from metric import correlation_analysis
from metric import scoring_function
from metric import window_metrics
import temporal_dataset


# ============================================================
# 1. Configuration
# ============================================================

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

MODEL_LIFE_WINDOW = 10
N_WINDOWS = 170

INITIAL_BALANCE = 10_000.0
TP_GAIN = 0.04
FP_LOSS = 0.03
THRESHOLD = 0.5
CORR_THRESHOLD = 0.98
TOP_N_CORRELATED_PAIRS = 10
SCORE_MODEL_PATH = Path("artifacts/score_function.json")
TEMPORAL_MODE = "stochastic"
STOCHASTIC_RANDOM_STATE = 42
STOCHASTIC_JITTER = 0.22
STOCHASTIC_SMOOTH_WINDOW = 48

INITIAL_TRAIN_SIZE = 97  # 1797 - 170*10 = 97


# ============================================================
# 2. Chargement dataset
#    Digits -> binaire
#    "momentum" = chiffre >= 5
# ============================================================

temporal_data = temporal_dataset.load_temporal_digits(
    mode=TEMPORAL_MODE,
    random_state=STOCHASTIC_RANDOM_STATE,
    jitter=STOCHASTIC_JITTER,
    smooth_window=STOCHASTIC_SMOOTH_WINDOW,
)

X = temporal_data["data"]
y = temporal_data["momentum"]
temporal_stats = temporal_data["sequence_stats"]

n_samples = len(y)
required = INITIAL_TRAIN_SIZE + N_WINDOWS * MODEL_LIFE_WINDOW

if n_samples < required:
    raise ValueError(
        f"Dataset trop petit : {n_samples} lignes, "
        f"il en faut au moins {required}."
    )

print(f"Nombre total de lignes : {n_samples}")
print(f"Mode temporel         : {temporal_data['mode']}")
print(f"Taux momentum         : {temporal_stats['momentum_rate']:.4f}")
print(f"Switches temporels    : {temporal_stats['switches']}")
print(f"Train initial         : {INITIAL_TRAIN_SIZE}")
print(f"Fenêtres              : {N_WINDOWS}")
print(f"Taille fenêtre        : {MODEL_LIFE_WINDOW}")
print()


# ============================================================
# 3. Modele
# ============================================================

def train_model(X_train, y_train, seed=42):
    model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


# ============================================================
# 4. Split initial random
# ============================================================

all_indices = np.arange(n_samples)
rng.shuffle(all_indices)

train_indices = all_indices[:INITIAL_TRAIN_SIZE].tolist()
pool_indices = all_indices[INITIAL_TRAIN_SIZE:].tolist()

X_train = X[train_indices]
y_train = y[train_indices]

print(f"Taille train initial : {len(train_indices)}")
print(f"Taille pool restante : {len(pool_indices)}")
print()


# ============================================================
# 5. Boucle séquentielle par fenêtres
# ============================================================

balance = INITIAL_BALANCE
rows = []

for window_id in range(1, N_WINDOWS + 1):
    chosen_pos = rng.choice(len(pool_indices), size=MODEL_LIFE_WINDOW, replace=False)
    chosen_pos_set = set(chosen_pos.tolist())

    window_indices = [pool_indices[pos] for pos in chosen_pos]

    pool_indices = [idx for j, idx in enumerate(pool_indices) if j not in chosen_pos_set]

    X_window = X[window_indices]
    y_window = y[window_indices]

    model = train_model(X_train, y_train, seed=RANDOM_STATE + window_id)

    y_proba = model.predict_proba(X_window)[:, 1]

    metric_row = window_metrics.build_window_metric_dict(
        y_window,
        y_proba,
        balance_start=balance,
        threshold=THRESHOLD,
        tp_gain=TP_GAIN,
        fp_loss=FP_LOSS,
    )
    balance = metric_row["balance_end"]

    row = {
        "window_id": window_id,
        "train_size_before_update": len(X_train),
        "test_window_size": MODEL_LIFE_WINDOW,
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
calibration_cols = window_metrics.CALIBRATION_COLS
decision_cols = window_metrics.DECISION_COLS
threshold_geometry_cols = window_metrics.THRESHOLD_GEOMETRY_COLS
gain_cols = window_metrics.GAIN_COLS

M_classification = df_windows[classification_cols].values
M_separation = df_windows[separation_cols].values
M_calibration = df_windows[calibration_cols].values
M_decision = df_windows[decision_cols].values
M_threshold_geometry = df_windows[threshold_geometry_cols].values
M_gain = df_windows[gain_cols].values

print("=== Formes des vecteurs par famille ===")
print("classification      :", M_classification.shape)
print("separation          :", M_separation.shape)
print("calibration         :", M_calibration.shape)
print("decision            :", M_decision.shape)
print("threshold_geometry  :", M_threshold_geometry.shape)
print("gain                :", M_gain.shape)
print()


# ============================================================
# 7. Matrice globale des métriques
# ============================================================

metric_cols = window_metrics.ALL_METRIC_COLS

X_metrics = df_windows[metric_cols].values

print("=== Aperçu des fenêtres ===")
print(df_windows.head())
print()

print("=== Matrice globale des métriques ===")
print(df_windows[metric_cols].head())
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

for j, name in enumerate(metric_cols):
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

non_constant_indices = [
    j for j in range(X_metrics_clean.shape[1])
    if np.std(X_metrics_clean[:, j]) > 1e-12
]

selected_indices = []
current = None

for j in non_constant_indices:
    col = X_metrics_clean[:, [j]]

    if current is None:
        selected_indices.append(j)
        current = col
        continue

    rank_before = np.linalg.matrix_rank(current)
    candidate = np.column_stack([current, col])
    rank_after = np.linalg.matrix_rank(candidate)

    if rank_after > rank_before:
        selected_indices.append(j)
        current = candidate

selected_metric_names = [metric_cols[j] for j in selected_indices]

print("=== Base empirique gloutonne ===")
print("Indices   :", selected_indices)
print("Métriques :", selected_metric_names)
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

print("=== Corrélations avec gain_effective ===")
corr_with_gain = (
    df_windows[metric_cols]
    .corr(numeric_only=True)["gain_effective"]
    .sort_values(ascending=False)
)
print(corr_with_gain)
print()


# ============================================================
# 13. Matrice de corrélation complète
# ============================================================

corr_df = correlation_analysis.correlation_matrix_ignore_nan(
    X_metrics,
    metric_cols,
)
print("=== Corrélation entre métriques ===")
print(corr_df.round(3))
print()


# ============================================================
# 14. Paires fortement corrélées
# ============================================================

print(f"=== Paires de métriques fortement corrélées (>=|{CORR_THRESHOLD}|) ===")
correlated_pairs = correlation_analysis.extract_correlated_pairs(
    corr_df,
    threshold=CORR_THRESHOLD,
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

print(f"=== Top {TOP_N_CORRELATED_PAIRS} paires les plus corrélées ===")
for pair in correlated_pairs[:TOP_N_CORRELATED_PAIRS]:
    print(
        f"{pair['left']:25s} <-> {pair['right']:25s} : "
        f"corr={pair['corr']:+.6f} abs={pair['abs_corr']:.6f}"
    )
print()


# ============================================================
# 15. Construction d'une base empirique EX ANTE
#     (sans les métriques de gain)
# ============================================================

ex_ante_metric_cols = window_metrics.EX_ANTE_METRIC_COLS

X_ex_ante = df_windows[ex_ante_metric_cols].values
X_ex_ante_clean = np.nan_to_num(X_ex_ante, nan=0.0)

print("=== Matrice ex ante utilisée pour construire la base ===")
print(pd.DataFrame(X_ex_ante_clean, columns=ex_ante_metric_cols).head())
print()

rank_ex_ante = np.linalg.matrix_rank(X_ex_ante_clean)
print(f"Rang de la matrice ex ante : {rank_ex_ante}")
print(f"Nombre de métriques ex ante: {X_ex_ante_clean.shape[1]}")
print()

non_constant_ex_ante_indices = [
    j for j in range(X_ex_ante_clean.shape[1])
    if np.std(X_ex_ante_clean[:, j]) > 1e-12
]

selected_ex_ante_indices = []
current_ex_ante = None

for j in non_constant_ex_ante_indices:
    col = X_ex_ante_clean[:, [j]]

    if current_ex_ante is None:
        selected_ex_ante_indices.append(j)
        current_ex_ante = col
        continue

    rank_before = np.linalg.matrix_rank(current_ex_ante)
    candidate = np.column_stack([current_ex_ante, col])
    rank_after = np.linalg.matrix_rank(candidate)

    if rank_after > rank_before:
        selected_ex_ante_indices.append(j)
        current_ex_ante = candidate

basis_ex_ante_cols = [ex_ante_metric_cols[j] for j in selected_ex_ante_indices]

print("=== Base empirique ex ante ===")
print("Indices   :", selected_ex_ante_indices)
print("Métriques :", basis_ex_ante_cols)
print()

if current_ex_ante is not None:
    print("=== Matrice associée à la base ex ante ===")
    print(pd.DataFrame(current_ex_ante, columns=basis_ex_ante_cols).head())
    print()


# ============================================================
# 16. Apprentissage et sauvegarde du score
# ============================================================

target_gain = df_windows["gain_effective"].values.astype(float)

score_model, score_values = scoring_function.fit_linear_score_model(
    df_windows[basis_ex_ante_cols].values,
    basis_ex_ante_cols,
    target_gain,
    target_name="gain_effective",
)
basis_ex_ante_cols_kept = score_model["feature_names"]
Z_basis = scoring_function.standardize_with_score_model(
    df_windows[basis_ex_ante_cols_kept].values,
    score_model,
)
w = np.asarray(score_model["weights"], dtype=float)
score_gain_corr = float(score_model["training_target_corr"])

saved_score_path = scoring_function.save_score_model(
    score_model,
    SCORE_MODEL_PATH,
    metadata={
        "random_state": RANDOM_STATE,
        "model_life_window": MODEL_LIFE_WINDOW,
        "n_windows": N_WINDOWS,
        "initial_balance": INITIAL_BALANCE,
        "tp_gain": TP_GAIN,
        "fp_loss": FP_LOSS,
        "threshold": THRESHOLD,
        "initial_train_size": INITIAL_TRAIN_SIZE,
    },
)

print("=== Poids du score appris sur la base ex ante ===")
for name, weight in zip(basis_ex_ante_cols_kept, w):
    print(f"{name:25s} : {weight:+.6f}")
print()

print("=== Base ex ante après centrage-réduction ===")
print("Métriques gardées :", basis_ex_ante_cols_kept)
print("Forme de Z_basis  :", Z_basis.shape)
print()

print("=== Corrélation Pearson(score, gain_effective) ===")
print(score_gain_corr)
print()

print("=== Score sauvegardé ===")
print(saved_score_path.resolve())
print()


# ============================================================
# 17. Ajouter le score au DataFrame
# ============================================================

df_windows["score_metric"] = score_values

print("=== Aperçu score_metric vs gain_effective ===")
print(df_windows[["window_id", "score_metric", "gain_effective"]].head())
print()

print("=== Corrélation finale score_metric / gain_effective ===")
print(
    df_windows[["score_metric", "gain_effective"]]
    .corr(numeric_only=True)
    .iloc[0, 1]
)
print()


# ============================================================
# 18. Corrélations du score avec toutes les métriques
# ============================================================

print("=== Corrélations avec score_metric ===")
metric_plus_score_cols = metric_cols + ["score_metric"]
corr_with_score = (
    df_windows[metric_plus_score_cols]
    .corr(numeric_only=True)["score_metric"]
    .sort_values(ascending=False)
)
print(corr_with_score)
print()


# ============================================================
# 19. Affichage des vecteurs constituant la base
# ============================================================

print("=== Vecteurs de la base empirique (colonnes) ===")

if current is not None:
    for idx, (name, col) in enumerate(zip(selected_metric_names, current.T)):
        print(f"\nVecteur {idx+1} : {name}")
        print(col)
else:
    print("Aucune base disponible.")

print("\n=== Vecteurs de base après centrage-réduction ===")

if Z_basis is not None:
    for idx, (name, col) in enumerate(zip(basis_ex_ante_cols_kept, Z_basis.T)):
        print(f"\nVecteur normalisé {idx+1} : {name}")
        print(col)
else:
    print("Aucune base normalisée disponible.")
