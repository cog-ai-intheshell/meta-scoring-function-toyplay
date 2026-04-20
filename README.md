# Meta Scoring Function

Projet d'experimentation autour d'une fonction de score "meta" pour evaluer des fenetres de predictions binaires.

Le pipeline actuel repose sur :

- un CSV complet `data_gathering/dataset.csv`
- un CSV de developpement `data_gathering/dataset_dev.csv`
- un CSV holdout `data_gathering/dataset_holdout.csv`
- un backtest sequentiel sur `120` fenetres de developpement de taille `10`
- un modele `XGBoost`
- une base de score organisee en `4` familles de metriques
- un score hierarchique appris sur les `120` fenetres du dataset de developpement
- une evaluation finale sur `50` fenetres holdout jamais vues

## Objectif

Le projet ne cherche pas seulement a predire `momentum`.
Il cherche surtout a construire un score de qualite de fenetre aligne avec `gain_effective`.

En pratique :

- une prediction positive correcte rapporte `+4%`
- une prediction positive incorrecte coute `-3%`
- une prediction negative n'ouvre pas de trade

Le score meta ne contient pas les metriques de gain dans sa base, mais il est appris pour expliquer au mieux `gain_effective`.

## Dataset

Le pipeline manipule maintenant trois CSV :

- `data_gathering/dataset.csv` : la sequence temporelle complete
- `data_gathering/dataset_dev.csv` : le train initial + les `120` premieres fenetres
- `data_gathering/dataset_holdout.csv` : les `50` dernieres fenetres

Ce fichier contient :

- `dataset_index`
- `original_index`
- les colonnes `pixel_*`
- la cible binaire `momentum`

Le CSV est construit a partir du dataset `digits`, puis reordonne pour suivre une structure de regimes inspiree de `data_gathering/reference.csv`.

Le dossier `data_gathering` contient donc maintenant :

- `digits_source.csv` : la source fixe du dataset `digits` avec `pixel_*` et `number_label`
- `reference.csv` : la reference de regimes
- `dataset.csv` : le dataset complet
- `dataset_dev.csv` : le split de developpement
- `dataset_holdout.csv` : le split holdout

## Configuration

Toute la configuration est centralisee dans `config.py`.

On y retrouve notamment :

- les chemins des donnees et artefacts
- `DIGIT_POSITIVE_MIN_LABEL = 5`
- `INITIAL_TRAIN_SIZE = 97`
- `MODEL_LIFE_WINDOW = 10`
- `N_WINDOWS = 120`
- `FULL_PROTOCOL_WINDOWS = 170`
- `SCORE_TRAIN_WINDOWS = 120`
- `HOLDOUT_WINDOWS = 50`
- les parametres de gain
- les bornes de recherche Optuna
- le seuil de filtrage des correlations fortes
- `TARGET_CORR_METHOD`
- `REDUNDANCY_CORR_METHOD`
- le type d'operateur lineaire utilise pour apprendre les sous-scores et le score final

Le changement d'operateur se fait simplement dans `config.py` via :

- `SCORE_OPERATOR_TYPE`
- `SCORE_OPERATOR_RIDGE_LAMBDA`

Aujourd'hui, la valeur recommandee dans le projet est :

- `SCORE_OPERATOR_TYPE = "marginal_corr"`

Le parametre `SCORE_OPERATOR_RIDGE_LAMBDA` n'est utile que si `SCORE_OPERATOR_TYPE = "ridge"`.

Pour la generation du CSV :

- si `DIGIT_POSITIVE_MIN_LABEL = 5`, alors `momentum = 1` pour les labels `5..9`
- si `DIGIT_POSITIVE_MIN_LABEL = 7`, alors `momentum = 1` pour les labels `7..9`
L'idee est que toute modification experimentale passe par `config.py`.

## Pipeline principal

Le script principal est `main.py`.

Il execute les etapes suivantes :

1. Charge `data_gathering/dataset_dev.csv`.
2. Utilise les `97` premieres lignes comme train initial.
3. Parcourt ensuite `120` fenetres contigues de `10` observations.
4. Entraine un `XGBClassifier` avant chaque fenetre.
5. Calcule les metriques realisees de fenetre.
6. Decoupe la base de score en `4` familles :
   - `classification`
   - `separation`
   - `generalization`
   - `stabilite`
7. Sur les `120` fenetres du dataset de developpement, construit une base par famille :
   - retrait des colonnes constantes
   - retrait des colonnes trop correlees
   - priorite donnee a la colonne la plus corrélée a `gain_effective`
   - selection gloutonne par rang
8. Apprend un sous-score lineaire dans chaque famille.
9. Construit une matrice des `4` sous-scores :
   - `score_classification`
   - `score_separation`
   - `score_generalization`
   - `score_stabilite`
10. Apprend le score final sur cette matrice `120 x 4`.
11. Sauvegarde le modele dans `artifacts/score_function.json`.

Important :

- les metriques utilisees dans la base du score sont des metriques **realisees sur la fenetre courante**
- elles ne sont donc pas "ex ante"
- le holdout final n'est plus visible dans `main.py`

## Base de score actuelle

La base finale est hierarchique.

### Famille `classification`

- `tp`
- `tn`
- `fp`
- `fn`
- `precision`
- `recall`
- `specificity`
- `mcc`
- `balanced_accuracy`

### Famille `separation`

- `auc`
- `logloss`
- `brier`
- `median_dist_tp`
- `median_dist_tn`
- `median_dist_fp`
- `median_dist_fn`
- `threshold_confidence_ratio`
- `threshold_confidence_gap`

### Famille `generalization`

- `empirical_overfitting`
- `empirical_bias`
- `empirical_variance`
- `empirical_residual_noise`

### Famille `stabilite`

- `auc_std`
- `logloss_std`
- `recall_std`
- `precision_std`
- `ppr_std`
- `threshold_std`
- `threshold_drift`
- `robustness_gap`
- `stability_score`

Les metriques de gain restent calculees, mais elles ne font pas partie de la base de score.

Quand une fenetre ne contient aucune opportunite (`gain_max_possible = 0`) :

- `gain_effective = 1.0` si aucun trade n'est pris
- `gain_effective = 0.0` si l'on atteint la pire perte cumulative possible en faux positifs
- entre les deux, `gain_effective` varie continument selon la perte reellement subie

## Apprentissage du score

L'apprentissage du score se fait dans `metric/scoring_function.py`.

Le modele de score n'est plus un simple score lineaire "plat".
Le pipeline est maintenant en deux etages :

1. un sous-score lineaire par famille
2. un score final lineaire sur les `4` sous-scores de famille

Autrement dit :

- chaque famille produit un resume numerique
- la base finale du score vaut :
  - `base_finale = [score_classification, score_separation, score_generalization, score_stabilite]`

Le JSON sauvegarde contient donc :

- les modeles de famille
- les bases retenues dans chaque famille
- le modele final
- la correlation d'entrainement avec `gain_effective`
- les metadonnees de configuration

## Operateurs lineaires disponibles

Le projet permet de changer facilement d'operateur lineaire sans modifier le reste du pipeline.

Ce choix s'applique :

- aux sous-scores de famille
- au score final appris sur les `4` sous-scores

Le changement se fait uniquement dans `config.py`.

### Operateur `marginal_corr`

Configuration :

- `SCORE_OPERATOR_TYPE = "marginal_corr"`

Principe :

- chaque poids est la correlation marginale configurable entre une coordonnee standardisee de la base et la cible standardisee
- le vecteur des poids est ensuite normalise

La methode de correlation utilisee est celle definie dans `config.py` via :

- `TARGET_CORR_METHOD = "pearson"` ou `"spearman"`

Formellement, si `Z` est la base standardisee et `y` la cible standardisee :

- `w_j = corr(Z_j, y)`
- `w = w / ||w||`

Interet :

- simple
- interpretable
- stable
- dans l'etat actuel du projet, c'est l'operateur qui generalise le mieux sur le holdout

### Operateur `ridge`

Configuration :

- `SCORE_OPERATOR_TYPE = "ridge"`
- `SCORE_OPERATOR_RIDGE_LAMBDA = 1.0` par exemple

Principe :

- on apprend les poids par regression ridge sur les coordonnees standardisees

Formule :

- `w_lambda = (Z^T Z + lambda I)^-1 Z^T y`

Interet :

- utilise l'information multivariee complete
- gere mieux les bases corrélées
- permet de regulariser l'operateur

Limite observee ici :

- dans nos derniers essais, il collait mieux au train mais generalisait moins bien sur le holdout que `marginal_corr`

### Recommandation actuelle

Pour l'instant, la version a garder comme reference est :

- `SCORE_OPERATOR_TYPE = "marginal_corr"`

et `ridge` reste disponible pour les experimentations futures.

## Recherche d'hyperparametres

Le script `optuna_xgb_simple.py` recharge la score function et l'utilise comme objectif d'optimisation sur le dataset de developpement uniquement.

Le principe est :

- rejouer le backtest fenetre par fenetre
- recalculer toutes les metriques
- reappliquer le score hierarchique sauvegarde
- optimiser uniquement sur les `120` fenetres du dataset de developpement
- retourner la mediane des scores de fenetre comme objectif Optuna

Le holdout n'est plus evalue dans ce script.

Le meilleur resultat est sauvegarde dans `artifacts/optuna_best_xgb.json`.

## Evaluation holdout finale

Le script `evaluate_holdout_xgb.py` sert a evaluer le meilleur XGB sur les `50` fenetres jamais vues.

Il :

- recharge la score function
- recharge les meilleurs hyperparametres trouves par Optuna
- recharge `dataset_dev.csv` et `dataset_holdout.csv`
- rejoue le protocole complet en concaténant les deux splits
- isole les fenetres `121..170`
- affiche le resume global :
  - `TP`, `TN`, `FP`, `FN`
  - `gain_realized`, `gain_ratio`, `gain_effective_mean`
  - `score_median`, `score_mean`, `score_corr_gain_effective`
- affiche le detail par fenetre
- recalcule en interne les sous-scores de famille pour construire le score final holdout
- genere un plot des probabilites predites du XGB
- genere aussi un plot du referentiel holdout
- ajoute une ligne d'historique dans le CSV des runs holdout

Le plot est produit via `ploting.py`.

## Structure du projet

```text
.
├── backtest.py
├── config.py
├── data_gathering/
│   ├── digits_source.csv
│   ├── dataset.csv
│   ├── dataset_dev.csv
│   ├── dataset_holdout.csv
│   ├── generate_dataset.py
│   ├── prebuilt_dataset.py
│   └── reference.csv
├── evaluate_holdout_xgb.py
├── main.py
├── optuna_xgb_simple.py
├── ploting.py
├── metric/
│   ├── algebra.py
│   ├── basis_transforms.py
│   ├── classification_metrics.py
│   ├── correlation_analysis.py
│   ├── decision_metrics.py
│   ├── gain_metrics.py
│   ├── generalization_metrics.py
│   ├── scoring_function.py
│   └── window_metrics.py
├── tools/
│   └── name_generator.py
└── artifacts/
    ├── holdout_evaluation.json
    ├── holdout_model_runs.csv
    ├── holdout_probability_predictions.png
    ├── holdout_referential_score_evolution.png
    ├── referential_score_evolution.png
    ├── score_function.json
    └── optuna_best_xgb.json
```

## Installation

Le depot ne contient pas encore de `requirements.txt` ou `pyproject.toml`.
Les dependances minimales visibles dans le code sont :

- `matplotlib`
- `numpy`
- `pandas`
- `xgboost`
- `optuna`

Installation rapide :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib numpy pandas xgboost optuna
```

## Execution

Ordre d'execution recommande :

```bash
python3 main.py
python3 optuna_xgb_simple.py
python3 evaluate_holdout_xgb.py
```

Ce flux donne :

- `main.py` : apprentissage de la score function hierarchique
- `optuna_xgb_simple.py` : optimisation XGB sur les `120` fenetres train
- `evaluate_holdout_xgb.py` : evaluation finale sur les `50` fenetres holdout + plot

Regenerer le CSV preconstruit :

```bash
python3 -m data_gathering.generate_dataset \
  --digits-csv data_gathering/digits_source.csv
```

Par defaut, le generateur utilise :

- `data_gathering/digits_source.csv` comme source `digits`
- `data_gathering/reference.csv` comme template de regimes
- `data_gathering/dataset.csv` comme sortie complete
- `data_gathering/dataset_dev.csv` comme split de developpement
- `data_gathering/dataset_holdout.csv` comme split holdout

Tu peux aussi surcharger les chemins si besoin :

```bash
python3 -m data_gathering.generate_dataset \
  --digits-csv data_gathering/digits_source.csv \
  --template-csv data_gathering/reference.csv \
  --output data_gathering/dataset.csv \
  --output-dev data_gathering/dataset_dev.csv \
  --output-holdout data_gathering/dataset_holdout.csv
```

## Artefacts produits

### `artifacts/score_function.json`

Contient :

- les sous-modeles de famille
- la base retenue dans chaque famille
- le modele final sur les `4` sous-scores
- les metadonnees de configuration

### `artifacts/optuna_best_xgb.json`

Contient :

- la meilleure valeur de score
- les meilleurs hyperparametres XGBoost
- le nombre d'essais
- un resume train du meilleur modele sur le dataset de developpement

### `artifacts/holdout_evaluation.json`

Contient :

- le resume global holdout
- le detail par fenetre
- le nom du modele evalue
- les chemins des plots generes
- le rappel du CSV d'historique mis a jour

### `artifacts/holdout_probability_predictions.png`

Plot des probabilites predites par le meilleur XGB sur le holdout :

- points colores selon la classe reelle
- marquage `TP`, `TN`, `FP`, `FN`
- seuil de decision

### `artifacts/holdout_referential_score_evolution.png`

Trajectoire des fenetres holdout dans le referentiel des `4` sous-scores :

- axes spatiaux : `classification`, `separation`, `generalization`
- couleur : `stabilite`
- score final derive ensuite de ces coordonnees

### `artifacts/referential_score_evolution.png`

Trajectoire equivalente sur le dataset de developpement appris dans `main.py`.

### `artifacts/holdout_model_runs.csv`

Historique cumulatif des evaluations holdout.

Chaque ligne contient notamment :

- le nom du modele evalue
- les valeurs de configuration actives
- les hyperparametres XGB
- `gain_realized`, `gain_ratio`, `score_median`
- `tp`, `tn`, `fp`, `fn`

## Points d'attention

### Sequence temporelle

Le projet utilise un CSV preconstruit dont l'ordre encode deja la dynamique temporelle.
Les scripts respectent strictement cet ordre :

- train initial sur les `97` premieres lignes
- apprentissage du score sur les `120` fenetres du CSV de developpement
- optimisation XGB sur ces memes `120` fenetres
- evaluation finale sur les `50` dernieres fenetres du CSV holdout

### Sens de `train`, `test` et `all`

Dans `main.py`, tout se passe sur le dataset de developpement :

- `train` = fenetres `1..120`
- il n'y a plus de `test` interne jamais vu

Le vrai critere de generalisation est maintenant uniquement le holdout final evalue par `evaluate_holdout_xgb.py`.

### Modules utilitaires

Le coeur du pipeline courant passe surtout par :

- `backtest.py`
- `metric/window_metrics.py`
- `metric/scoring_function.py`
- `metric/correlation_analysis.py`

## Resume rapide

Si tu veux comprendre le depot vite :

- `config.py` est la source unique de configuration
- `backtest.py` porte la boucle sequentielle commune
- `main.py` apprend un score hierarchique en `4` familles sur `dataset_dev.csv`
- `optuna_xgb_simple.py` optimise XGBoost sur les `120` fenetres du dataset de developpement
- `evaluate_holdout_xgb.py` juge le meilleur modele sur les `50` fenetres jamais vues
