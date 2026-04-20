# Meta Scoring Function

Projet d'experimentation autour d'une fonction de score "meta" pour evaluer des fenetres de predictions binaires.

L'idee generale est la suivante :

- on prend le dataset `digits` de `scikit-learn`
- on le transforme en probleme binaire : `digit >= 5` devient `momentum = 1`
- on entraine un modele `XGBoost`
- on evalue ce modele sur une succession de fenetres
- on calcule, pour chaque fenetre, un ensemble de metriques de classification, calibration, decision et gain
- on apprend ensuite une fonction de score lineaire capable d'expliquer au mieux le `gain_effective`

Le score appris est sauvegarde dans `artifacts/score_function.json` et peut ensuite etre reutilise pour piloter une recherche d'hyperparametres avec Optuna.

## Objectif

Le depot ne cherche pas seulement a mesurer la qualite predictive d'un modele.
Il cherche surtout a construire une metrique composee qui soit bien alignee avec une performance de type "gain" sur une fenetre.

En pratique :

- une prediction positive correcte rapporte `+4%`
- une prediction positive incorrecte coute `-3%`
- une prediction negative n'ouvre pas de trade

Le projet apprend donc un score "ex ante" a partir de metriques observables avant d'utiliser la cible de gain.

## Pipeline principal

Le script principal est `main.py`.

Il execute les etapes suivantes :

1. Charge le dataset `digits` puis construit une cible binaire `momentum`.
2. Genere une version temporelle du dataset via `temporal_dataset.py`.
3. Initialise un train set puis un pool d'exemples restants.
4. Repete une boucle sur `170` fenetres de `10` observations.
5. Entraine un `XGBClassifier` a chaque fenetre.
6. Calcule les metriques de la fenetre :
   - confusion (`tp`, `tn`, `fp`, `fn`)
   - classification (`precision`, `recall`, `specificity`, `mcc`)
   - separation (`auc`)
   - calibration (`logloss`, `brier`)
   - decision (`ppr`, `prevalence`)
   - geometrie autour du seuil (`dist_tp`, `dist_fp`, `dist_tn`, `dist_fn`)
   - gain (`gain_realized`, `gain_max_possible`, `gain_effective`, `balance_start`, `balance_end`)
7. Analyse la redondance entre metriques via le rang matriciel et les correlations.
8. Construit une base empirique gloutonne sur les metriques "ex ante" seulement.
9. Apprend une fonction de score lineaire sur `gain_effective`.
10. Sauvegarde le modele de score dans `artifacts/score_function.json`.

## Apprentissage du score

L'apprentissage du score se fait dans `metric/scoring_function.py`.

Le principe est simple :

- les metriques selectionnees sont centrees-reduites
- les colonnes constantes ou quasi constantes sont eliminees
- chaque feature recueille un poids proportionnel a sa correlation de Pearson avec la cible
- le vecteur de poids est normalise
- le score final d'une fenetre est le produit matriciel `Z @ w`

Le fichier JSON sauvegarde contient :

- les noms de features gardees
- les moyennes et ecarts-types de standardisation
- les poids du score
- la correlation d'entrainement entre le score et `gain_effective`
- des metadonnees de configuration

## Recherche d'hyperparametres

Le script `optuna_xgb_simple.py` recharge le score appris et l'utilise comme objectif d'optimisation.

Au lieu d'optimiser directement l'accuracy ou le logloss, il :

- rejoue le pipeline par fenetres
- recalcule les metriques pour un jeu d'hyperparametres donne
- applique la fonction de score sauvegardee
- retourne la mediane des scores de fenetre comme objectif Optuna

Le meilleur resultat est sauvegarde dans `artifacts/optuna_best_xgb.json`.

## Structure du projet

```text
.
├── main.py
├── temporal_dataset.py
├── optuna_xgb_simple.py
├── metric/
│   ├── classification_metrics.py
│   ├── correlation_analysis.py
│   ├── decision_metrics.py
│   ├── gain_metrics.py
│   ├── generalization_metrics.py
│   ├── scoring_function.py
│   ├── structural_scores.py
│   └── window_metrics.py
└── artifacts/
    ├── score_function.json
    └── optuna_best_xgb.json
```

## Installation

Le depot ne contient pas encore de fichier `requirements.txt` ou `pyproject.toml`.
Les dependances minimales visibles dans le code sont :

- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `optuna`

Installation rapide :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn xgboost optuna
```

## Execution

Generer la fonction de score :

```bash
python3 main.py
```

Lancer la recherche d'hyperparametres apres generation du score :

```bash
python3 optuna_xgb_simple.py
```

## Artefacts produits

### `artifacts/score_function.json`

Contient la fonction de score apprise :

- features retenues
- statistiques de standardisation
- poids lineaires
- metadonnees de configuration

### `artifacts/optuna_best_xgb.json`

Contient le resultat de la recherche Optuna :

- meilleure valeur de score
- meilleurs hyperparametres XGBoost
- nombre d'essais

## Points d'attention

### Sequence temporelle

Le module `temporal_dataset.py` peut construire une sequence "stochastic" avec une dynamique temporelle synthetique.
Cependant, dans `main.py`, les indices sont ensuite melanges puis les fenetres sont tirees aleatoirement dans un pool restant.

Il faut donc voir ce pipeline comme :

- une simulation sequentielle par fenetres
- avec accumulation progressive du train set
- mais pas comme un backtest strictement chronologique

### Modules utilitaires

Les modules `metric/structural_scores.py` et `metric/generalization_metrics.py` servent surtout de bibliotheques utilitaires pour des scores plus riches, mais ils ne sont pas au coeur du pipeline execute par `main.py`.

## Resume rapide

Si tu veux comprendre le depot vite :

- `main.py` apprend une fonction de score sur des fenetres de prediction
- `temporal_dataset.py` fabrique la version pseudo-temporelle du dataset
- `metric/window_metrics.py` assemble toutes les metriques de fenetre
- `metric/scoring_function.py` apprend et sauvegarde le score lineaire
- `optuna_xgb_simple.py` optimise XGBoost avec ce score comme objectif
