# Mathematiques du projet

Ce document formalise le pipeline actif du depot avec des ecritures mathematiques simples, lisibles a la fois dans GitHub et dans Obsidian.

Le point important est que la fonction de score actuelle est **hierarchique** :

1. un sous-score par famille de metriques
2. un score final appris sur les `4` sous-scores

Les fichiers centraux sont :

- `config.py`
- `data_gathering/generate_dataset.py`
- `data_gathering/prebuilt_dataset.py`
- `data_gathering/digits_source.csv`
- `data_gathering/reference.csv`
- `data_gathering/dataset.csv`
- `data_gathering/dataset_dev.csv`
- `data_gathering/dataset_holdout.csv`
- `main.py`
- `optuna_xgb_simple.py`
- `evaluate_holdout_xgb.py`
- `metric/classification_metrics.py`
- `metric/decision_metrics.py`
- `metric/gain_metrics.py`
- `metric/generalization_metrics.py`
- `metric/window_metrics.py`
- `metric/scoring_function.py`
- `metric/correlation_analysis.py`

## 1. Objet du projet

Le projet apprend :

1. un classifieur binaire `XGBoost` qui predit une probabilite de `momentum`
2. une fonction de score de fenetre qui essaie d'ordonner les fenetres selon leur qualite economique

Le flux conceptuel est :

$$
\text{pixels}
\longrightarrow
\hat{p}
\longrightarrow
\hat{y}
\longrightarrow
\phi_t
\longrightarrow
\text{sous-scores de famille}
\longrightarrow
\text{score final}.
$$

Ici :

- $\hat{p}$ est une probabilite predite
- $\hat{y}$ est la decision binaire au seuil
- $\phi_t$ est le vecteur de metriques de la fenetre $t$

## 2. Dataset fixe

Le generateur construit trois CSV :

- `data_gathering/dataset.csv` : sequence complete
- `data_gathering/dataset_dev.csv` : train initial + `120` premieres fenetres
- `data_gathering/dataset_holdout.csv` : `50` dernieres fenetres

On note la sequence complete :

$$
\mathcal{D} = \{(x_t, y_t)\}_{t=1}^n
$$

et les deux splits physiques :

$$
\mathcal{D}_{\text{dev}} = \{(x_t, y_t)\}_{t=1}^{1297},
\qquad
\mathcal{D}_{\text{holdout}} = \{(x_t, y_t)\}_{t=1298}^{1797}.
$$

avec :

$$
n = 1797,
\qquad
x_t \in \mathbb{R}^{64},
\qquad
y_t \in \{0,1\}.
$$

Les colonnes utilisees comme features sont les `64` colonnes `pixel_*`.

La cible binaire est `momentum`.

## 3. Construction du CSV "digits melania style"

Le CSV final provient de deux ingredients :

1. le dataset `digits` stocke dans `data_gathering/digits_source.csv`
2. la suite binaire `momentum` du fichier `data_gathering/reference.csv`

### 3.1 Source `digits`

On part d'observations :

$$
\mathcal{D}_{\text{digits}} = \{(u_i, z_i)\}_{i=1}^n
$$

avec :

$$
u_i \in \mathbb{R}^{64},
\qquad
z_i \in \{0,1,\dots,9\}.
$$

La cible binaire associee est configurable via `config.DIGIT_POSITIVE_MIN_LABEL`.
Si on note ce seuil $\kappa$, alors :

$$
y_i^{\text{digits}} = \mathbf{1}[z_i \ge \kappa].
$$

Exemples :

- si $\kappa = 5$, alors les chiffres `5..9` sont positifs
- si $\kappa = 7$, alors les chiffres `7..9` sont positifs

On note :

$$
n_1 = \sum_{i=1}^n y_i^{\text{digits}},
\qquad
n_0 = n - n_1.
$$

### 3.2 Template MELANIA

On lit la colonne `momentum` et on obtient une suite :

$$
m_1, m_2, \dots, m_T \in \{0,1\}.
$$

On la decompose en runs :

$$
[(v_1,\ell_1), (v_2,\ell_2), \dots, (v_K,\ell_K)],
$$

ou :

$$
v_k \in \{0,1\},
\qquad
\ell_k \ge 1.
$$

### 3.3 Reallocation des longueurs

Le but est de reproduire une structure de regimes semblable, mais avec les effectifs de `digits`.

On construit de nouvelles longueurs :

$$
\tilde{\ell}_1, \dots, \tilde{\ell}_K
$$

telles que :

$$
\sum_{k : v_k = 0} \tilde{\ell}_k = n_0,
\qquad
\sum_{k : v_k = 1} \tilde{\ell}_k = n_1,
\qquad
\tilde{\ell}_k \ge 1.
$$

Le code :

1. reserve `1` element par run
2. distribue le reliquat proportionnellement
3. affecte les restes par parties fractionnaires decroissantes

On obtient ainsi une suite cible :

$$
\tilde{y}_1, \tilde{y}_2, \dots, \tilde{y}_n.
$$

### 3.4 Reordonnancement des exemples

On separe les indices `digits` en deux pools :

$$
P = \{i : y_i^{\text{digits}} = 1\},
\qquad
N = \{i : y_i^{\text{digits}} = 0\}.
$$

Puis on parcourt $\tilde{y}_t$ :

- si $\tilde{y}_t = 1$, on prend le prochain element de $P$
- si $\tilde{y}_t = 0$, on prend le prochain element de $N$

On obtient une permutation $\rho$ telle que :

$$
x_t = u_{\rho(t)},
\qquad
y_t = \tilde{y}_t.
$$

## 4. Protocole sequentiel par fenetres

Les constantes de `config.py` sont :

$$
\texttt{INITIAL\_TRAIN\_SIZE} = 97,
\qquad
\texttt{MODEL\_LIFE\_WINDOW} = 10,
\qquad
\texttt{N\_WINDOWS} = 120,
\qquad
\texttt{FULL\_PROTOCOL\_WINDOWS} = 170.
$$

On a :

$$
97 + 120 \times 10 = 1297
\quad \text{pour } \mathcal{D}_{\text{dev}},
\qquad
97 + 170 \times 10 = 1797
\quad \text{pour le protocole complet}.
$$

Le dataset de developpement consomme donc :

- soit dans le train initial
- soit dans une unique fenetre d'evaluation dev

### 4.1 Fenetre $t$

On note :

$$
T_0 = \{1,\dots,97\}.
$$

Pour le protocole complet, chaque fenetre $t \in \{1,\dots,170\}$ est definie par :

$$
W_t =
\left\{
97 + 10(t-1) + 1,
\dots,
97 + 10t
\right\}.
$$

Le train disponible avant la fenetre $t$ est :

$$
T_{t-1} = \{1,\dots,97 + 10(t-1)\}.
$$

Apres observation :

$$
T_t = T_{t-1} \cup W_t = \{1,\dots,97 + 10t\}.
$$

### 4.2 Modele probabiliste

A chaque fenetre $t$, on entraine un classifieur sur $T_{t-1}$ :

$$
f_t : \mathbb{R}^{64} \to [0,1].
$$

Pour chaque observation $i \in W_t$ :

$$
\hat{p}_i^{(t)} = f_t(x_i).
$$

La decision binaire utilise le seuil fixe :

$$
\tau = 0.5,
\qquad
\hat{y}_i^{(t)} = \mathbf{1}\!\left[\hat{p}_i^{(t)} \ge \tau\right].
$$

## 5. Split apprentissage / holdout

Le projet distingue :

1. apprentissage de la fonction de score
2. optimisation Optuna
3. evaluation holdout finale

Le point important est que le holdout est maintenant **physiquement separe** dans un autre CSV :

- `main.py` ne charge que `dataset_dev.csv`
- `optuna_xgb_simple.py` ne charge que `dataset_dev.csv`
- `evaluate_holdout_xgb.py` est le seul script qui recharge aussi `dataset_holdout.csv`

Les constantes sont :

$$
\texttt{SCORE\_TRAIN\_WINDOWS} = 120,
\qquad
\texttt{HOLDOUT\_WINDOWS} = 50.
$$

On note :

$$
\mathcal{W}_{\text{train}} = \{W_1,\dots,W_{120}\},
\qquad
\mathcal{W}_{\text{holdout}} = \{W_{121},\dots,W_{170}\}.
$$

La regle de protocole est :

- la fonction de score est apprise uniquement sur $\mathcal{W}_{\text{train}}$
- Optuna optimise XGB uniquement sur $\mathcal{W}_{\text{train}}$
- l'evaluation finale observe $\mathcal{W}_{\text{holdout}}$

## 6. Metriques de classification

Dans une fenetre donnee, on omet l'indice $t$.
La taille de fenetre vaut :

$$
m = 10.
$$

On dispose de :

$$
y_i \in \{0,1\},
\qquad
\hat{p}_i \in [0,1],
\qquad
\hat{y}_i \in \{0,1\},
\qquad i = 1,\dots,m.
$$

### 6.1 Comptages de confusion

$$
\mathrm{TP} = \sum_{i=1}^m \mathbf{1}[y_i = 1 \land \hat{y}_i = 1],
$$

$$
\mathrm{TN} = \sum_{i=1}^m \mathbf{1}[y_i = 0 \land \hat{y}_i = 0],
$$

$$
\mathrm{FP} = \sum_{i=1}^m \mathbf{1}[y_i = 0 \land \hat{y}_i = 1],
$$

$$
\mathrm{FN} = \sum_{i=1}^m \mathbf{1}[y_i = 1 \land \hat{y}_i = 0].
$$

### 6.2 Convention `safe_divide`

Le code utilise souvent :

$$
\operatorname{safe\_divide}(a,b) =
\begin{cases}
0 & \text{si } b = 0, \\
\dfrac{a}{b} & \text{sinon}.
\end{cases}
$$

### 6.3 Metriques retenues

La famille `classification` contient :

$$
\operatorname{Precision}
=
\frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}},
$$

$$
\operatorname{Recall}
=
\frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}},
$$

$$
\operatorname{Specificity}
=
\frac{\mathrm{TN}}{\mathrm{TN} + \mathrm{FP}},
$$

$$
\operatorname{MCC}
=
\frac{\mathrm{TP}\mathrm{TN} - \mathrm{FP}\mathrm{FN}}
{\sqrt{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})(\mathrm{TN}+\mathrm{FP})(\mathrm{TN}+\mathrm{FN})}},
$$

$$
\operatorname{BalancedAccuracy}
=
\frac{\operatorname{Recall} + \operatorname{Specificity}}{2}.
$$

## 7. Metriques de separation

### 7.1 Probabilistes

Avant calcul du log-loss, les probabilites sont clippees :

$$
\tilde{p}_i = \min\bigl(\max(\hat{p}_i,\varepsilon), 1-\varepsilon\bigr),
$$

ou $\varepsilon$ est l'epsilon machine.

Le log-loss vaut :

$$
\operatorname{LogLoss}
=
-\frac{1}{m}
\sum_{i=1}^m
\left(
y_i \log(\tilde{p}_i)
+
(1-y_i)\log(1-\tilde{p}_i)
\right).
$$

Le Brier score vaut :

$$
\operatorname{Brier}
=
\frac{1}{m}\sum_{i=1}^m (\hat{p}_i - y_i)^2.
$$

L'AUC est interpretee comme :

$$
\operatorname{AUC}
=
\mathbb{P}\!\left(\hat{p}^+ > \hat{p}^-\right)
+
\frac{1}{2}\mathbb{P}\!\left(\hat{p}^+ = \hat{p}^-\right).
$$

Si une fenetre ne contient qu'une seule classe, le code fixe :

$$
\operatorname{AUC} = \operatorname{NaN}.
$$

### 7.2 Geometrie autour du seuil

On pose la distance au seuil :

$$
d_i = |\hat{p}_i - \tau|.
$$

Le code retient les **medians** de distance :

$$
\operatorname{median\_dist\_tp}
=
\operatorname{median}\{d_i : y_i=1,\hat{y}_i=1\},
$$

$$
\operatorname{median\_dist\_tn}
=
\operatorname{median}\{d_i : y_i=0,\hat{y}_i=0\},
$$

$$
\operatorname{median\_dist\_fp}
=
\operatorname{median}\{d_i : y_i=0,\hat{y}_i=1\},
$$

$$
\operatorname{median\_dist\_fn}
=
\operatorname{median}\{d_i : y_i=1,\hat{y}_i=0\}.
$$

Si l'ensemble est vide, le code renvoie `0.0`.

Le code calcule aussi :

$$
\operatorname{threshold\_confidence\_ratio}
=
\frac{\operatorname{mean}(d_i \mid \hat{y}_i = y_i)}
{\operatorname{mean}(d_i \mid \hat{y}_i \ne y_i)},
$$

$$
\operatorname{threshold\_confidence\_gap}
=
\operatorname{mean}(d_i \mid \hat{y}_i = y_i)
-
\operatorname{mean}(d_i \mid \hat{y}_i \ne y_i).
$$

Une quantite auxiliaire non utilisee directement dans la base est aussi calculee :

$$
\operatorname{implicit\_threshold},
$$

qui aligne au mieux le taux de positifs predits sur la prevalence observee.

## 8. Metriques de generalization

La famille `generalization` n'est pas construite par validation croisee stricte.
Dans l'etat actuel du code, ce sont des **proxys sequentiels** calcules a partir de l'historique des fenetres deja vues et du train courant.

On note :

$$
\ell_t^{\text{val}} = \operatorname{LogLoss}(W_t),
\qquad
\ell_t^{\text{train}} = \operatorname{LogLoss}(T_{t-1}).
$$

Sur l'historique disponible, le code utilise :

$$
\operatorname{empirical\_overfitting}
=
\operatorname{mean}(\ell^{\text{val}})
-
\operatorname{mean}(\ell^{\text{train}}),
$$

$$
\operatorname{empirical\_bias}
=
\operatorname{mean}(\ell^{\text{val}}),
$$

$$
\operatorname{empirical\_variance}
=
\operatorname{var}(\ell^{\text{val}}),
$$

$$
\operatorname{empirical\_residual\_noise}
=
\operatorname{var}(y_i - \hat{p}_i)
\quad \text{sur la fenetre courante}.
$$

## 9. Metriques de stabilite

La famille `stabilite` est construite a partir des metriques historiques deja observees.

Le code calcule notamment :

$$
\operatorname{auc\_std} = \operatorname{std}(\operatorname{AUC}),
$$

$$
\operatorname{logloss\_std} = \operatorname{std}(\operatorname{LogLoss}),
$$

$$
\operatorname{recall\_std} = \operatorname{std}(\operatorname{Recall}),
$$

$$
\operatorname{precision\_std} = \operatorname{std}(\operatorname{Precision}),
$$

$$
\operatorname{ppr\_std} = \operatorname{std}(\operatorname{PPR}),
$$

$$
\operatorname{threshold\_std} = \operatorname{std}(\operatorname{implicit\_threshold}),
$$

$$
\operatorname{threshold\_drift}
=
\max(\operatorname{implicit\_threshold})
-
\min(\operatorname{implicit\_threshold}),
$$

$$
\operatorname{robustness\_gap}
=
\operatorname{mean}(\operatorname{MCC})
-
\min(\operatorname{MCC}),
$$

$$
\operatorname{stability\_score}
=
\frac{1}{1 + \operatorname{std}(\operatorname{MCC})}.
$$

## 10. Modele economique

Le projet interprete une prediction positive comme l'ouverture d'un trade.

Les constantes sont :

$$
g_{\text{TP}} = 0.04,
\qquad
\ell_{\text{FP}} = 0.03.
$$

Si $\hat{y}_i = 1$, alors :

$$
r_i =
\begin{cases}
g_{\text{TP}} & \text{si } y_i = 1, \\
-\ell_{\text{FP}} & \text{si } y_i = 0.
\end{cases}
$$

Si $\hat{y}_i = 0$, aucun trade n'est execute.

La serie des rendements executes est :

$$
(r_i)_{i \in I},
\qquad
I = \{i : \hat{y}_i = 1\}.
$$

Le rendement realise de la fenetre vaut :

$$
R_{\text{real}} = \prod_{i \in I}(1+r_i) - 1.
$$

Si le bilan initial vaut $B_{\text{start}}$ :

$$
B_{\text{end}} = B_{\text{start}}(1 + R_{\text{real}}),
$$

$$
G_{\text{real}} = B_{\text{end}} - B_{\text{start}}.
$$

On note le nombre de vrais positifs potentiels :

$$
n_+ = \sum_{i=1}^m \mathbf{1}[y_i = 1].
$$

Le gain maximal possible est :

$$
B_{\max} = B_{\text{start}}(1 + g_{\text{TP}})^{n_+},
\qquad
G_{\max} = B_{\max} - B_{\text{start}}.
$$

La cible economique de la fenetre est :

$$
\operatorname{gain\_effective}
=
\begin{cases}
\dfrac{G_{\text{real}}}{G_{\max}} & \text{si } G_{\max} \ne 0, \\
1 & \text{si } G_{\max} = 0 \text{ et aucun trade n'est execute}, \\
0 & \text{si } G_{\max} = 0 \text{ et au moins un trade est execute}.
\end{cases}
$$

## 11. Vecteur de metriques par fenetre

Pour chaque fenetre $W_t$, `metric/window_metrics.py` construit un dictionnaire.

Les familles principales de la base sont :

- `classification`
- `separation`
- `generalization`
- `stabilite`

On peut voir le vecteur complet d'une fenetre comme :

$$
\phi_t \in \mathbb{R}^q.
$$

Sur l'ensemble des `170` fenetres :

$$
\Phi =
\begin{bmatrix}
\phi_1^T \\
\phi_2^T \\
\vdots \\
\phi_{170}^T
\end{bmatrix}.
$$

## 12. Base de score par famille

La base n'est plus construite globalement sur toutes les colonnes a la fois.

Pour chaque famille $F \in \{\text{classification}, \text{separation}, \text{generalization}, \text{stabilite}\}$, on construit une matrice train :

$$
X_F \in \mathbb{R}^{120 \times p_F}.
$$

### 12.1 Retrait des colonnes constantes

Une colonne $j$ est d'abord retiree si :

$$
\operatorname{std}(X_{F,\cdot j}) \le \varepsilon.
$$

### 12.2 Filtrage des colonnes trop correlees

Pour chaque colonne restante $c_j$, on calcule sa correlation a la cible train :

$$
r_j = \operatorname{corr}(c_j, g),
$$

ou :

$$
g =
\begin{bmatrix}
\operatorname{gain\_effective}(\phi_1) \\
\vdots \\
\operatorname{gain\_effective}(\phi_{120})
\end{bmatrix}.
$$

On calcule aussi la matrice de corrélation paire a paire.

Si deux colonnes $c_i$ et $c_j$ verifient :

$$
|\operatorname{corr}(c_i, c_j)| \ge \rho_{\max},
$$

avec $\rho_{\max} = \texttt{CORR\_THRESHOLD}$, alors on garde celle qui a la plus forte correlation absolue a la cible :

$$
\text{garder } c_i
\Longleftrightarrow
|r_i| \ge |r_j|.
$$

Le filtrage n'est donc plus arbitraire.

### 12.3 Selection gloutonne par rang

Sur les colonnes restantes, on construit une base gloutonne :

$$
C_j \text{ est retenue }
\Longleftrightarrow
\operatorname{rang}([B \mid C_j]) > \operatorname{rang}(B).
$$

On obtient ainsi, dans chaque famille, une sous-base :

$$
\mathcal{B}_F = \{f_1^{(F)}, \dots, f_{d_F}^{(F)}\}.
$$

## 13. Sous-score de famille

Pour chaque famille $F$, on apprend un score lineaire sur la matrice :

$$
X_F^\star \in \mathbb{R}^{120 \times d_F}.
$$

### 13.1 Standardisation

Pour chaque colonne $j$ :

$$
\mu_j^{(F)} = \operatorname{nanmean}(X_{F,\cdot j}^\star),
\qquad
\sigma_j^{(F)} = \operatorname{nanstd}(X_{F,\cdot j}^\star).
$$

La matrice standardisee vaut :

$$
Z_{F,ij} = \frac{X_{F,ij}^\star - \mu_j^{(F)}}{\sigma_j^{(F)}}.
$$

### 13.2 Operateur lineaire du sous-score

Pour chaque famille, on standardise aussi la cible train :

$$
g_i^{(z)} = \frac{g_i - \mu_g}{\sigma_g}.
$$

Puis on construit un operateur lineaire a partir des correlations marginales entre chaque coordonnee standardisee et la cible standardisee :

$$
w_{F,j}^{\text{raw}} = \operatorname{corr}(Z_{F,\cdot j}, g^{(z)}).
$$

On normalise ensuite ce vecteur de poids :

$$
w_F = \frac{w_F^{\text{raw}}}{\|w_F^{\text{raw}}\|_2}.
$$

Le sous-score de famille vaut alors :

$$
s_F = Z_F w_F.
$$

Autrement dit, le code actif utilise une combinaison lineaire heuristique guidee par les correlations marginales a la cible.

Concretement, le code produit :

- `score_classification`
- `score_separation`
- `score_generalization`
- `score_stabilite`

## 14. Score final

Les `4` sous-scores definissent une nouvelle matrice :

$$
S =
\begin{bmatrix}
s_{\text{classification}} &
s_{\text{separation}} &
s_{\text{generalization}} &
s_{\text{stabilite}}
\end{bmatrix}
\in \mathbb{R}^{120 \times 4}.
$$

Le score final est appris sur cette matrice exactement avec la meme logique.

Si l'on note $Z_S$ la version standardisee de $S$, alors chaque composante brute du vecteur de poids vaut :

$$
\alpha_j^{\text{raw}} = \operatorname{corr}(Z_{S,\cdot j}, g^{(z)}),
$$

puis :

$$
\alpha = \frac{\alpha^{\text{raw}}}{\|\alpha^{\text{raw}}\|_2}.
$$

Si l'on note :

$$
\alpha =
\begin{bmatrix}
\alpha_1 \\
\alpha_2 \\
\alpha_3 \\
\alpha_4
\end{bmatrix},
$$

alors le score final d'une fenetre vaut :

$$
s = Z_S \alpha.
$$

Autrement dit :

$$
\operatorname{score\_final}
=
\alpha_1 z_{\text{classification}}
+
\alpha_2 z_{\text{separation}}
+
\alpha_3 z_{\text{generalization}}
+
\alpha_4 z_{\text{stabilite}}.
$$

Les quantites $z_{\text{classification}}, z_{\text{separation}}, z_{\text{generalization}}, z_{\text{stabilite}}$
sont les coordonnees standardisees de la fenetre dans la base finale, avant application de l'operateur lineaire.

Le fichier `artifacts/score_function.json` sauvegarde :

- les modeles de famille
- les variables retenues dans chaque famille
- le modele final sur les `4` sous-scores

## 15. Signification de `train` et `all`

Dans `main.py` :

- `train` = fenetres `1..120`
- `all` = les memes fenetres `1..120` du dataset de developpement

Mathématiquement :

$$
\operatorname{corr}_{\text{train}}
=
\operatorname{corr}(s_t, g_t)
\quad \text{pour } t=1,\dots,120,
$$

$$
\operatorname{corr}_{\text{all}}
=
\operatorname{corr}(s_t, g_t)
\quad \text{pour } t=1,\dots,120.
$$

Le holdout n'apparait plus dans `main.py`.
Le critere principal de generalisation est l'evaluation finale realisee par `evaluate_holdout_xgb.py` sur $\mathcal{W}_{\text{holdout}}$.

## 16. Objectif Optuna

Le script `optuna_xgb_simple.py` recharge la fonction de score gelee.

Pour un jeu d'hyperparametres $\theta$, on rejoue les `120` fenetres train et on calcule :

$$
s_t(\theta)
\quad \text{pour } t = 1,\dots,120.
$$

L'objectif Optuna est :

$$
J(\theta)
=
\operatorname{median}\bigl(s_1(\theta), \dots, s_{120}(\theta)\bigr).
$$

Ici :

- $s_t(\theta)$ designe le `score_metric` de la fenetre $t$
- l'objectif Optuna est donc bien la mediane des scores de fenetre
- il ne s'agit pas d'une moyenne
- il ne s'agit pas non plus d'une mediane de `gain_ratio`

Optuna ne maximise donc pas directement :

- ni l'accuracy
- ni l'AUC
- ni `gain_effective`
- ni `gain_ratio`

Il maximise la mediane du score meta appris auparavant.

Le holdout n'est plus evalue dans `optuna_xgb_simple.py`.

## 17. Evaluation holdout finale

Le script `evaluate_holdout_xgb.py` recharge :

- la fonction de score gelee
- les meilleurs hyperparametres XGB
- `dataset_dev.csv`
- `dataset_holdout.csv`

Puis il rejoue tout le processus sequentiel et n'analyse finalement que :

$$
W_{121}, \dots, W_{170}.
$$

Il calcule notamment :

$$
\mathrm{TP}_{\text{holdout}} = \sum_{t=121}^{170} \mathrm{TP}_t,
$$

$$
\mathrm{TN}_{\text{holdout}} = \sum_{t=121}^{170} \mathrm{TN}_t,
$$

$$
\mathrm{FP}_{\text{holdout}} = \sum_{t=121}^{170} \mathrm{FP}_t,
$$

$$
\mathrm{FN}_{\text{holdout}} = \sum_{t=121}^{170} \mathrm{FN}_t.
$$

Le ratio de gain holdout affiche vaut :

$$
\operatorname{gain\_ratio}_{\text{holdout}}
=
\frac{\sum_{t=121}^{170} G_{\text{real}}^{(t)}}
{\sum_{t=121}^{170} G_{\max}^{(t)}}
\quad \text{si le denominateur est non nul.}
$$

Il faut insister sur le fait que :

- `gain_ratio` n'est pas une mediane
- `gain_ratio` n'est pas une moyenne de ratios par fenetre
- `gain_ratio` est un ratio global agrege, obtenu comme quotient des sommes

Autrement dit :

$$
\operatorname{gain\_ratio}_{\text{holdout}}
\neq
\operatorname{median}\left(
\frac{G_{\text{real}}^{(121)}}{G_{\max}^{(121)}},
\dots,
\frac{G_{\text{real}}^{(170)}}{G_{\max}^{(170)}}
\right).
$$

Le script calcule aussi :

$$
\operatorname{score\_median}_{\text{holdout}}
=
\operatorname{median}(s_{121}, \dots, s_{170}),
$$

$$
\operatorname{score\_mean}_{\text{holdout}}
=
\frac{1}{50}\sum_{t=121}^{170} s_t,
$$

ainsi que :

$$
\operatorname{gain\_effective\_mean}_{\text{holdout}}
=
\frac{1}{50}\sum_{t=121}^{170} \operatorname{gain\_effective}_t,
$$

et :

$$
\operatorname{corr}(s_t, \operatorname{gain\_effective}_t)
\quad \text{pour } t = 121,\dots,170.
$$

## 18. Visualisation des probabilites

Le module `ploting.py` cree un nuage de points pour les `500` observations du holdout.

Pour chaque observation $k$ du holdout, on trace :

$$
(k,\hat{p}_k).
$$

Le graphe ajoute :

- une ligne horizontale au seuil $\tau = 0.5$
- une coloration des points selon la verite terrain
- un marquage explicite des `TP`, `TN`, `FP`, `FN`

## 19. Conventions d'implementation importantes

Pour bien lire les resultats, il faut garder les conventions du code :

1. `safe_divide(a, 0) = 0`
2. `AUC = NaN` si une fenetre ne contient qu'une seule classe
3. les moyennes ou medianes sur ensemble vide valent `0.0`
4. la fonction de score est apprise sur les `120` premieres fenetres seulement
5. Optuna optimise sur ces memes `120` fenetres
6. les `50` dernieres fenetres jouent le role de holdout final

## 20. Resume mathematique compact

Le pipeline complet peut se resumer ainsi :

1. construction d'une suite binaire $\tilde{y}_1,\dots,\tilde{y}_n$ de style MELANIA
2. reordonnancement des exemples `digits` pour obtenir $(x_t, y_t)$
3. backtest sequentiel sur `170` fenetres contigues
4. calcul, pour chaque fenetre, d'un vecteur de metriques $\phi_t$
5. decoupage de la base en `4` familles
6. apprentissage d'un sous-score dans chaque famille
7. apprentissage d'un score final sur la matrice des `4` sous-scores
8. optimisation Optuna de XGB sur la mediane des scores de fenetres train
9. evaluation finale sur `50` fenetres holdout jamais utilisees pour apprendre le score ou choisir les hyperparametres

Si on note $\theta^\star$ les meilleurs hyperparametres XGB et $\alpha^\star$ les poids du score final, l'objet observe sur le holdout est :

$$
\left\{
\left(
s_t(\theta^\star, \alpha^\star),
\operatorname{gain\_effective}_t(\theta^\star)
\right)
\right\}_{t=121}^{170}.
$$

Autrement dit, le projet ne cherche pas seulement a predire une classe binaire; il cherche a construire un critere de qualite de fenetre aligne avec la performance economique.
