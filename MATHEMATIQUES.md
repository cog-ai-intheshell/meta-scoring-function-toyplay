# Mathematiques du projet

Ce document formalise la mathematique implemente dans le depot.
Il suit le code au plus pres, y compris quand une convention d'implementation differe d'une definition "academique" plus standard.

Le pipeline effectif est porte principalement par :

- `main.py`
- `temporal_dataset.py`
- `metric/window_metrics.py`
- `metric/classification_metrics.py`
- `metric/decision_metrics.py`
- `metric/gain_metrics.py`
- `metric/scoring_function.py`
- `metric/correlation_analysis.py`
- `optuna_xgb_simple.py`

Les modules `metric/structural_scores.py` et `metric/generalization_metrics.py` sont aussi formalises plus bas, mais ils ne sont pas au coeur du pipeline execute par `main.py`.

## 1. Notations globales

On note le dataset original :

$$
\mathcal{D} = \{(x_i, z_i)\}_{i=1}^n
$$

avec :

$$
x_i \in \mathbb{R}^{64}, \qquad z_i \in \{0,1,\dots,9\}, \qquad n = 1797.
$$

La cible binaire est definie par :

$$
y_i = \mathbf{1}[z_i \ge 5].
$$

On note :

$$
n_1 = \sum_{i=1}^n y_i, \qquad n_0 = n - n_1.
$$

La prevalence globale est :

$$
\pi = \frac{1}{n}\sum_{i=1}^n y_i.
$$

Dans tout le projet, les tableaux sont traites numeriquement avec `numpy`.

## 2. Construction du dataset "temporel"

Le module `temporal_dataset.py` propose deux modes :

- `original`
- `stochastic`

### 2.1 Mode original

En mode `original`, le dataset est simplement :

$$
X = (x_1,\dots,x_n), \qquad y = (y_1,\dots,y_n).
$$

L'indice temporel est alors l'ordre original du dataset `digits`.

### 2.2 Mode stochastic

Le mode `stochastic` ne modifie pas les exemples bruts, mais reconstruit un ordre synthetique dans lequel la sequence binaire des labels suit une dynamique lisse aleatoire.

#### 2.2.1 Bruit lisse normalise

On genere un bruit gaussien :

$$
\varepsilon_t \sim \mathcal{N}(0,1), \qquad t = 1,\dots,n.
$$

Si la fenetre de lissage vaut `smooth_window = h > 1`, on applique une convolution par noyau uniforme :

$$
k_u = \frac{1}{h}, \qquad u=1,\dots,h,
$$

et

$$
\tilde{\varepsilon} = \varepsilon * k.
$$

Le signal est ensuite normalise par sa norme infinie :

$$
s_t = \frac{\tilde{\varepsilon}_t}{\max_u |\tilde{\varepsilon}_u|},
$$

avec la convention suivante :

$$
\max_u |\tilde{\varepsilon}_u| < 10^{-12} \implies s_t = 0 \quad \forall t.
$$

Ainsi :

$$
s_t \in [-1,1].
$$

#### 2.2.2 Poids de tirage temporels

Le code construit ensuite des poids bruts :

$$
a_t = \mathrm{clip}\left(\pi + \gamma s_t,\ p_{\min},\ p_{\max}\right),
$$

ou :

- $\gamma$ est `jitter`
- $p_{\min}$ est `min_probability`
- $p_{\max}$ est `max_probability`

Puis :

$$
w_t = \frac{a_t}{\sum_{u=1}^n a_u}.
$$

Les quantites $w_t$ definissent une loi de probabilite sur les positions temporelles.

#### 2.2.3 Tirage des positions positives

On tire exactement $n_1$ positions parmi $\{1,\dots,n\}$, sans remise, avec probabilites proportionnelles a $w_t$.

Si $M \subset \{1,\dots,n\}$ est l'ensemble tire, avec $|M|=n_1$, alors la nouvelle sequence de labels est :

$$
\tilde{y}_t = \mathbf{1}[t \in M].
$$

Cette construction preserve exactement le nombre total de positifs :

$$
\sum_{t=1}^n \tilde{y}_t = n_1.
$$

#### 2.2.4 Reordonnancement des exemples

Le code separe ensuite les indices originaux en deux paquets :

$$
P = \{i : y_i = 1\}, \qquad N = \{i : y_i = 0\}.
$$

Il melange aleatoirement `P` et `N` independamment.
Puis il remplit la nouvelle chronologie :

- si $\tilde{y}_t = 1$, on prend le prochain indice melange de `P`
- si $\tilde{y}_t = 0$, on prend le prochain indice melange de `N`

On obtient donc une permutation $\rho$ telle que :

$$
\tilde{x}_t = x_{\rho(t)}, \qquad \tilde{z}_t = z_{\rho(t)}.
$$

La suite $(\tilde{x}_t,\tilde{y}_t)$ est la version "stochastic" du dataset.

### 2.3 Proprietes mathematiques du mode stochastic

Cette transformation :

- preserve le nombre total de positifs et de negatifs
- preserve les exemples bruts a permutation pres
- change la structure des runs temporels
- ne cree pas de nouvelle feature numerique
- ne change pas la definition de la cible binaire

En revanche, elle ne construit pas un veritable processus generatif temporel ; elle construit une chronologie synthetique avec controle partiel de la densite locale des positifs.

## 3. Protocole sequentiel par fenetres

Le script `main.py` travaille avec :

$$
\text{INITIAL\_TRAIN\_SIZE} = 97,
$$

$$
\text{MODEL\_LIFE\_WINDOW} = 10,
$$

$$
\text{N\_WINDOWS} = 170.
$$

Comme :

$$
97 + 170 \times 10 = 1797,
$$

toutes les observations du dataset sont consommees exactement une fois :

- soit dans le train initial
- soit dans une unique fenetre d'evaluation

### 3.1 Partition initiale

Le code melange d'abord tous les indices :

$$
\sigma \text{ permutation aleatoire de } \{1,\dots,n\}.
$$

Puis :

$$
T_0 = \{\sigma_1,\dots,\sigma_{97}\},
$$

$$
P_0 = \{\sigma_{98},\dots,\sigma_n\}.
$$

Ici :

- $T_0$ est l'ensemble d'entrainement initial
- $P_0$ est le pool restant

### 3.2 Fenetre $t$

Pour chaque fenetre $t \in \{1,\dots,170\}$ :

1. on tire uniformement sans remise un sous-ensemble

$$
W_t \subset P_{t-1}, \qquad |W_t| = 10
$$

2. on entraine un classifieur

$$
f_t : \mathbb{R}^{64} \to [0,1]
$$

sur le train courant $T_{t-1}$

3. on calcule, pour chaque $i \in W_t$, une probabilite

$$
p_i^{(t)} = f_t(x_i)
$$

4. on transforme la probabilite en decision binaire via le seuil fixe

$$
\tau = 0.5, \qquad \hat{y}_i^{(t)} = \mathbf{1}[p_i^{(t)} \ge \tau]
$$

5. on ajoute la fenetre au train :

$$
T_t = T_{t-1} \cup W_t
$$

6. on retire la fenetre du pool :

$$
P_t = P_{t-1} \setminus W_t
$$

On a donc :

$$
|T_t| = 97 + 10t,
$$

$$
|P_t| = 1700 - 10t.
$$

En particulier :

$$
P_{170} = \varnothing.
$$

### 3.3 Remarque importante

Le pipeline est sequentiel, mais il n'est pas strictement chronologique :

- `temporal_dataset.py` peut fabriquer une chronologie synthetique
- `main.py` remelange ensuite tous les indices
- les fenetres sont ensuite tirees aleatoirement dans le pool

On obtient donc une simulation sequentielle avec apprentissage cumulatif, mais pas un backtest temporel pur.

## 4. Modele de classification

Le modele entraine dans `main.py` est un `XGBClassifier` avec parametres fixes :

$$
\text{n\_estimators}=120,
$$

$$
\text{max\_depth}=4,
$$

$$
\text{learning\_rate}=0.08,
$$

$$
\text{subsample}=0.9,
$$

$$
\text{colsample\_bytree}=0.9.
$$

Mathematiquement, on peut simplement le voir comme une fonction :

$$
f_t(x) = \mathbb{P}(Y=1 \mid x; \theta_t),
$$

ou $\theta_t$ depend de l'echantillon d'entrainement au temps $t$.

Le detail interne du boosting n'est pas reimplemente dans le depot ; il est delegue a `xgboost`.

## 5. Metriques de classification

Le module `metric/classification_metrics.py` travaille sur une fenetre fixe de taille $m = 10$.
On omet l'indice de fenetre pour alleger l'ecriture.

### 5.1 Comptages de confusion

Pour une fenetre donnee, on note :

$$
\hat{y}_i \in \{0,1\}, \qquad y_i \in \{0,1\}.
$$

Les comptages sont :

$$
\mathrm{TP} = \sum_i \mathbf{1}[y_i=1 \land \hat{y}_i=1],
$$

$$
\mathrm{TN} = \sum_i \mathbf{1}[y_i=0 \land \hat{y}_i=0],
$$

$$
\mathrm{FP} = \sum_i \mathbf{1}[y_i=0 \land \hat{y}_i=1],
$$

$$
\mathrm{FN} = \sum_i \mathbf{1}[y_i=1 \land \hat{y}_i=0].
$$

### 5.2 Convention de division sure

Dans une grande partie du projet, le code utilise :

$$
\mathrm{safe\_divide}(a,b) =
\begin{cases}
0 & \text{si } b=0 \\
\frac{a}{b} & \text{sinon}
\end{cases}
$$

Cette convention est importante : plusieurs metriques valent donc `0.0` dans des cas degeneres ou une definition mathematique classique serait indeterminee.

### 5.3 Metriques de base

Les fonctions definies dans le module sont :

#### Accuracy

$$
\mathrm{Accuracy} = \frac{\mathrm{TP}+\mathrm{TN}}{\mathrm{TP}+\mathrm{TN}+\mathrm{FP}+\mathrm{FN}}.
$$

#### Precision

$$
\mathrm{Precision} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}.
$$

#### Recall

$$
\mathrm{Recall} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}.
$$

#### Specificity

$$
\mathrm{Specificity} = \frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}}.
$$

#### F1-score

$$
\mathrm{F1} =
\frac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}
{\mathrm{Precision} + \mathrm{Recall}}.
$$

#### Balanced accuracy

$$
\mathrm{BalancedAccuracy} =
\frac{\mathrm{Recall}+\mathrm{Specificity}}{2}.
$$

#### Matthews correlation coefficient

$$
\mathrm{MCC} =
\frac{\mathrm{TP}\cdot \mathrm{TN} - \mathrm{FP}\cdot \mathrm{FN}}
{\sqrt{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})(\mathrm{TN}+\mathrm{FP})(\mathrm{TN}+\mathrm{FN})}}.
$$

Si le denominateur vaut 0, le code renvoie `0.0`.

#### Predicted positive rate

$$
\mathrm{PPR} =
\frac{\mathrm{TP}+\mathrm{FP}}
{\mathrm{TP}+\mathrm{TN}+\mathrm{FP}+\mathrm{FN}}.
$$

#### Prevalence

$$
\mathrm{Prevalence} =
\frac{\mathrm{TP}+\mathrm{FN}}
{\mathrm{TP}+\mathrm{TN}+\mathrm{FP}+\mathrm{FN}}.
$$

#### Recall / prevalence ratio

$$
\mathrm{RecallPrevalenceRatio} =
\frac{\mathrm{Recall}}{\mathrm{Prevalence}}.
$$

Cette fonction existe dans le module mais n'est pas exploitee dans `main.py`.

## 6. Metriques probabilistes

Le meme module definit aussi des metriques sur les probabilites $p_i$.

### 6.1 Clipping numerique

Avant le calcul du log-loss, le code borne les probabilites :

$$
\tilde{p}_i = \min(\max(p_i, \varepsilon), 1-\varepsilon),
$$

ou $\varepsilon$ est l'epsilon machine du type `float`.

### 6.2 Log-loss

$$
\mathrm{LogLoss} =
-\frac{1}{m}\sum_{i=1}^m
\left(
y_i \log(\tilde{p}_i) + (1-y_i)\log(1-\tilde{p}_i)
\right).
$$

### 6.3 Baseline log-loss

Le module contient aussi une reference basee sur la prevalence :

$$
\pi_{\mathrm{fen}} = \frac{1}{m}\sum_{i=1}^m y_i,
$$

et

$$
\mathrm{BaselineLogLoss} =
-\left(
\pi_{\mathrm{fen}}\log(\pi_{\mathrm{fen}})
 + (1-\pi_{\mathrm{fen}})\log(1-\pi_{\mathrm{fen}})
\right).
$$

Si $\pi_{\mathrm{fen}} \in \{0,1\}$, le code renvoie `0.0`.

Cette metrique n'est pas utilisee directement dans `main.py`.

### 6.4 Brier score

$$
\mathrm{Brier} =
\frac{1}{m}\sum_{i=1}^m (p_i-y_i)^2.
$$

### 6.5 AUC

Le code implemente l'AUC ROC via les rangs moyens.
Soient :

- $m_+ = \sum_i \mathbf{1}[y_i=1]$
- $m_- = \sum_i \mathbf{1}[y_i=0]$

On trie les scores et on attribue des rangs moyens en cas d'egalite.
Si `rank_i` est le rang de $p_i$, alors :

$$
U =
\sum_{i:y_i=1} \mathrm{rank}_i
-
\frac{m_+(m_+ + 1)}{2}.
$$

L'AUC est :

$$
\mathrm{AUC} = \frac{U}{m_+ m_-}.
$$

Interpretation classique :

$$
\mathrm{AUC} =
\mathbb{P}(p^+ > p^-)
+
\frac{1}{2}\mathbb{P}(p^+ = p^-).
$$

#### Cas degenerate

Dans `probabilistic_metric_dict`, si une fenetre ne contient qu'une seule classe, alors le code ne calcule pas l'AUC et renvoie :

$$
\mathrm{AUC} = \mathrm{NaN}.
$$

Ce point est important pour la suite :

- la matrice de metriques contient potentiellement des `NaN`
- certaines etapes les ignorent
- d'autres les remplacent ensuite par `0.0`

## 7. Geometrie de decision autour du seuil

Le module `metric/decision_metrics.py` decrit la confiance relative a un seuil $\tau$.

### 7.1 Decision binaire

$$
\hat{y}_i = \mathbf{1}[p_i \ge \tau].
$$

### 7.2 Distance au seuil

Le code definit la marge de decision :

$$
d_i = |p_i - \tau|.
$$

Plus $d_i$ est grand, plus la prediction est eloignee du seuil.

### 7.3 Distances conditionnelles

Les metriques utilisees par `main.py` sont :

#### Distance moyenne des vrais positifs

$$
\mathrm{dist\_tp} =
\frac{1}{|\{i : y_i=1,\hat{y}_i=1\}|}
\sum_{i:y_i=1,\hat{y}_i=1} d_i.
$$

#### Distance moyenne des faux positifs

$$
\mathrm{dist\_fp} =
\frac{1}{|\{i : y_i=0,\hat{y}_i=1\}|}
\sum_{i:y_i=0,\hat{y}_i=1} d_i.
$$

#### Distance moyenne des vrais negatifs

$$
\mathrm{dist\_tn} =
\frac{1}{|\{i : y_i=0,\hat{y}_i=0\}|}
\sum_{i:y_i=0,\hat{y}_i=0} d_i.
$$

#### Distance moyenne des faux negatifs

$$
\mathrm{dist\_fn} =
\frac{1}{|\{i : y_i=1,\hat{y}_i=0\}|}
\sum_{i:y_i=1,\hat{y}_i=0} d_i.
$$

Si l'ensemble correspondant est vide, le code renvoie `0.0`.

### 7.4 Autres fonctions de seuil

Le module contient aussi d'autres objets mathematiques non utilises directement dans `main.py`.

#### Distance moyenne des decisions correctes

$$
\mathrm{threshold\_distance\_correct}
=
\frac{1}{|\{i:\hat{y}_i=y_i\}|}
\sum_{i:\hat{y}_i=y_i} d_i.
$$

#### Distance moyenne des decisions incorrectes

$$
\mathrm{threshold\_distance\_wrong}
=
\frac{1}{|\{i:\hat{y}_i\ne y_i\}|}
\sum_{i:\hat{y}_i\ne y_i} d_i.
$$

#### Ratio de confiance

$$
\mathrm{threshold\_confidence\_ratio}
=
\frac{\mathrm{threshold\_distance\_correct}}
{\mathrm{threshold\_distance\_wrong}}.
$$

#### Gap de confiance

$$
\mathrm{threshold\_confidence\_gap}
=
\mathrm{threshold\_distance\_correct}
-
\mathrm{threshold\_distance\_wrong}.
$$

#### Seuil implicite

Le module definit aussi un seuil $\tau_{\mathrm{impl}}$ qui aligne au mieux le taux de positifs predits sur la prevalence :

$$
\tau_{\mathrm{impl}}
=
\arg\min_{\tau \in \mathcal{C}}
\left|
\frac{1}{m}\sum_{i=1}^m \mathbf{1}[p_i \ge \tau]
-
\frac{1}{m}\sum_{i=1}^m y_i
\right|,
$$

ou l'ensemble de candidats est :

$$
\mathcal{C} = \{0,0.5,1\} \cup \{p_1,\dots,p_m\}.
$$

En cas d'egalite, le code choisit le seuil le plus proche de `0.5`.

#### Selection de seuil par F1

La fonction `top_threshold_candidates` trie les seuils candidats par :

1. score decroissant
2. distance croissante a un seuil par defaut
3. valeur de seuil croissante

Si aucune fonction de score custom n'est fournie, le score utilise est le F1-score.

## 8. Modele de gain

Le module `metric/gain_metrics.py` transforme les decisions positives en rendements financiers.

### 8.1 Rendement de trade

Le code ne considere comme trade execute que les observations pour lesquelles :

$$
\hat{y}_i = 1.
$$

Pour un trade execute, le rendement elementaire est :

$$
r_i =
\begin{cases}
g_{TP} & \text{si } \hat{y}_i=1 \text{ et } y_i=1 \\
-\ell_{FP} & \text{si } \hat{y}_i=1 \text{ et } y_i=0
\end{cases}
$$

avec, dans `main.py` :

$$
g_{TP} = 0.04, \qquad \ell_{FP} = 0.03.
$$

Les observations predites negatives n'entrent pas dans la serie de rendements.

### 8.2 Serie des rendements executes

Si $A = \{i : \hat{y}_i = 1\}$, alors :

$$
\mathcal{R} = (r_i)_{i \in A}.
$$

Sa longueur vaut :

$$
|\mathcal{R}| = \sum_i \mathbf{1}[\hat{y}_i = 1].
$$

### 8.3 Rendement cumulatif

Le rendement cumulatif multiplicatif sur la fenetre est :

$$
R_{\mathrm{cum}} =
\prod_{r \in \mathcal{R}} (1+r) - 1.
$$

Si aucun trade n'est execute, le code renvoie directement :

$$
R_{\mathrm{cum}} = 0.
$$

Ce comportement est coherent avec le produit vide.

### 8.4 Mise a jour de balance

Si la balance au debut de la fenetre vaut $B_{\mathrm{start}}$, alors :

$$
B_{\mathrm{end}} = B_{\mathrm{start}} (1 + R_{\mathrm{cum}}).
$$

Le gain realise vaut :

$$
G_{\mathrm{real}} = B_{\mathrm{end}} - B_{\mathrm{start}}.
$$

### 8.5 Gain maximal possible

Le nombre de momentum reels dans la fenetre vaut :

$$
n_{\mathrm{mom}} = \sum_i \mathbf{1}[y_i = 1].
$$

Le code suppose que le gain maximal est obtenu si tous les momentum reels sont captures avec succes, chacun rapportant $g_{TP}$ de maniere multiplicative :

$$
B_{\max} = B_{\mathrm{start}} (1 + g_{TP})^{n_{\mathrm{mom}}}.
$$

Donc :

$$
G_{\max} = B_{\max} - B_{\mathrm{start}}.
$$

### 8.6 Gain effectif

La metrique centrale est :

$$
G_{\mathrm{eff}} =
\frac{G_{\mathrm{real}}}{G_{\max}}.
$$

Mais le code traite le cas $G_{\max}=0$ ainsi :

$$
G_{\mathrm{eff}} =
\begin{cases}
1 & \text{si } G_{\max}=0 \text{ et aucun trade n'est execute} \\
0 & \text{si } G_{\max}=0 \text{ et au moins un trade est execute} \\
\frac{G_{\mathrm{real}}}{G_{\max}} & \text{sinon}
\end{cases}
$$

Cette convention penalise les faux trades dans une fenetre sans aucun vrai momentum.

### 8.7 Metriques supplementaires du module

Le module contient aussi d'autres definitions, utiles comme boite a outils :

#### Rendement moyen

$$
\mathrm{MeanReturn} = \frac{1}{m}\sum_{k=1}^m r_k.
$$

#### Volatilite empirique

$$
\mathrm{Volatility} = \sqrt{\frac{1}{m}\sum_{k=1}^m (r_k-\bar{r})^2}.
$$

Si un facteur d'annualisation $A$ est fourni :

$$
\mathrm{Volatility}_{\mathrm{ann}} = \sqrt{A}\,\mathrm{Volatility}.
$$

#### Volatilite baissiere

Pour un rendement acceptable minimal $r_{\min}$ :

$$
\mathrm{DownsideVol}
=
\sqrt{
\frac{1}{m}
\sum_{k=1}^m
\min(r_k-r_{\min},0)^2
}.
$$

#### Sharpe

$$
\mathrm{Sharpe}
=
\frac{\bar{r} - r_f}{\mathrm{Volatility}}.
$$

#### Sortino

$$
\mathrm{Sortino}
=
\frac{\bar{r} - r_{\min}}{\mathrm{DownsideVol}}.
$$

#### Max drawdown

En construisant une trajectoire de portefeuille :

$$
V_0 = V_{\mathrm{init}},
$$

$$
V_t = V_{\mathrm{init}}\prod_{k=1}^t (1+r_k),
$$

et

$$
P_t = \max_{u \le t} V_u,
$$

la drawdown vaut :

$$
D_t = \frac{P_t - V_t}{P_t},
$$

et la perte maximale est :

$$
\mathrm{MaxDrawdown} = \max_t D_t.
$$

#### Hit rate

$$
\mathrm{HitRate}
=
\frac{\#\{k : r_k > 0\}}{\#\{k\}}.
$$

ou avec `include_zero_as_win=True` :

$$
\mathrm{HitRate}
=
\frac{\#\{k : r_k \ge 0\}}{\#\{k\}}.
$$

#### Profit factor

$$
\mathrm{ProfitFactor}
=
\frac{\sum_k \max(r_k,0)}
{\sum_k \max(-r_k,0)}.
$$

Si les pertes brutes sont nulles :

- le code renvoie `inf` si les gains bruts sont strictement positifs
- sinon il renvoie `0.0`

#### Quantile de reference de marche

Pour une collection de gains de reference $M$ et un quantile $q$ :

$$
Q_q(M) = \mathrm{quantile}_q(M).
$$

#### Ratio effectif `r_eff_i`

$$
r_{\mathrm{eff},i}
=
\frac{G^{(i)}_{\mathrm{real}}}{Q_{0.75}(M_i)}.
$$

Si le denominateur est non fini ou trop petit, le code renvoie `NaN`.

## 9. Vecteur de metriques d'une fenetre

Le module `metric/window_metrics.py` assemble plusieurs familles de metriques.

### 9.1 Metriques ex ante

Le vecteur ex ante est constitue de :

$$
m_t \in \mathbb{R}^{17}
$$

avec les composantes :

$$
(\mathrm{tp},
\mathrm{tn},
\mathrm{fp},
\mathrm{fn},
\mathrm{precision},
\mathrm{recall},
\mathrm{specificity},
\mathrm{mcc},
\mathrm{auc},
\mathrm{logloss},
\mathrm{brier},
\mathrm{ppr},
\mathrm{prevalence},
\mathrm{dist\_tp},
\mathrm{dist\_fp},
\mathrm{dist\_tn},
\mathrm{dist\_fn}).
$$

On les appelle "ex ante" parce qu'elles ne contiennent pas explicitement les metriques de gain.

### 9.2 Metriques de gain

Le bloc de gain ajoute :

$$
g_t \in \mathbb{R}^{5}
$$

avec :

$$
(\mathrm{gain\_realized},
\mathrm{gain\_max\_possible},
\mathrm{gain\_effective},
\mathrm{balance\_start},
\mathrm{balance\_end}).
$$

### 9.3 Vecteur global

Le vecteur global vaut :

$$
u_t = (m_t, g_t) \in \mathbb{R}^{22}.
$$

Le dictionnaire de gain contient aussi :

- `n_true_momentum`
- `n_predicted_positive`

mais ces deux quantites ne sont pas incluses dans `ALL_METRIC_COLS`.

## 10. Matrices de metriques

Apres 170 fenetres, `main.py` construit :

$$
X_{\mathrm{metrics}} \in \mathbb{R}^{170 \times 22},
$$

et :

$$
X_{\mathrm{ex\_ante}} \in \mathbb{R}^{170 \times 17}.
$$

Les lignes correspondent aux fenetres, les colonnes aux metriques.

## 11. Analyse lineaire empirique

Le script effectue plusieurs diagnostics de structure.

### 11.1 Imputation locale des NaN pour le rang

Pour les calculs de rang matriciel, le code remplace :

$$
\mathrm{NaN} \mapsto 0.
$$

On note donc :

$$
\bar{X}_{\mathrm{metrics}} = \mathrm{nan\_to\_num}(X_{\mathrm{metrics}}, 0).
$$

et de meme pour $\bar{X}_{\mathrm{ex\_ante}}$.

### 11.2 Rang matriciel

Le rang empirique est :

$$
\mathrm{rank}(\bar{X}_{\mathrm{metrics}}).
$$

Si ce rang est strictement inferieur au nombre de colonnes, alors il existe une redondance lineaire empirique dans les metriques observees sur les 170 fenetres.

### 11.3 Colonnes constantes et quasi constantes

Pour chaque colonne $j$, le code calcule :

$$
s_j = \mathrm{std}(\bar{X}_{:,j}).
$$

Puis :

- si $s_j < 10^{-12}$ : colonne declaree constante
- si $10^{-12} \le s_j < 10^{-6}$ : colonne declaree quasi constante

### 11.4 Base empirique gloutonne

Le script construit ensuite une base empirique en parcourant les colonnes dans leur ordre d'apparition.

On initialise :

$$
S = \varnothing.
$$

Pour chaque colonne $j$ non constante, on ajoute $j$ a $S$ si et seulement si :

$$
\mathrm{rank}(\bar{X}_{:,S \cup \{j\}})
>
\mathrm{rank}(\bar{X}_{:,S}).
$$

Autrement dit, la colonne est garde uniquement si elle augmente le rang.

Cette procedure :

- produit une famille lineairement independante
- depend de l'ordre des colonnes
- n'est pas necessairement la base la plus interpretable
- n'est pas une selection optimale globale

Le meme algorithme est applique sur les seules metriques ex ante.

## 12. Correlations

Le module `metric/correlation_analysis.py` calcule des correlations de Pearson paire par paire en ignorant les `NaN`.

### 12.1 Correlation paire par paire avec masque de validite

Pour deux colonnes $x$ et $y$, on definit :

$$
\Omega = \{i : x_i \text{ et } y_i \text{ sont finis}\}.
$$

Si $|\Omega| < 2$, le code renvoie `NaN`.

Sinon :

$$
\rho(x,y)
=
\frac{
\sum_{i \in \Omega}(x_i-\bar{x}_{\Omega})(y_i-\bar{y}_{\Omega})
}{
\sqrt{
\sum_{i \in \Omega}(x_i-\bar{x}_{\Omega})^2
}
\sqrt{
\sum_{i \in \Omega}(y_i-\bar{y}_{\Omega})^2
}
}.
$$

Si l'ecart-type de $x$ ou $y$ sur $\Omega$ est inferieur a `eps`, le code renvoie aussi `NaN`.

### 12.2 Paires fortement correlees

Une paire $(j,k)$ est marquee "fortement correlee" si :

$$
|\rho_{jk}| \ge 0.98.
$$

Toutes les paires valides sont triees par correlation absolue decroissante.

## 13. Apprentissage de la fonction de score lineaire

Le coeur mathematique du projet est dans `metric/scoring_function.py`.

L'objectif est d'apprendre une fonction :

$$
S : \mathbb{R}^p \to \mathbb{R}
$$

qui mappe un vecteur de metriques ex ante vers un score cense etre aligne avec `gain_effective`.

### 13.1 Donnees d'entree

Le code prend :

- la base empirique ex ante selectionnee
- la cible

$$
t_i = \mathrm{gain\_effective}^{(i)}.
$$

Soit :

$$
X \in \mathbb{R}^{N \times p}, \qquad N=170.
$$

### 13.2 Standardisation des features

Pour chaque colonne $j$ :

$$
\mu_j = \mathrm{nanmean}(X_{:,j}),
$$

$$
\sigma_j = \mathrm{nanstd}(X_{:,j}).
$$

Le code conserve uniquement les colonnes telles que :

$$
\mu_j \text{ fini}, \qquad \sigma_j \text{ fini}, \qquad \sigma_j > \varepsilon.
$$

Les colonnes exclues sont donc :

- constantes
- quasi nulles
- ou numeriquement invalides

Pour les colonnes gardees :

$$
Z_{ij} = \frac{X_{ij}-\mu_j}{\sigma_j}.
$$

Puis :

$$
Z_{ij} = 0 \quad \text{si cette quantite est NaN}.
$$

Autrement dit, les `NaN` residuels sont remplaces par `0` apres centrage-reduction.

### 13.3 Standardisation de la cible

Le code calcule :

$$
\mu_t = \mathrm{nanmean}(t),
$$

$$
\sigma_t = \mathrm{nanstd}(t).
$$

Puis :

$$
z^{(t)}_i = \frac{t_i-\mu_t}{\sigma_t}.
$$

Si $\sigma_t < \varepsilon$ ou si $\sigma_t$ n'est pas fini, le code leve une erreur car l'optimisation est impossible.

### 13.4 Poids par correlations marginales

Pour chaque feature $j$, le poids brut est :

$$
a_j = \rho(Z_{:,j}, z^{(t)}),
$$

ou $\rho$ est ici une correlation de Pearson "safe" :

- si moins de 2 observations valides : `0.0`
- si l'ecart-type de l'une des deux variables est trop petit : `0.0`
- sinon correlation de Pearson classique

Le vecteur brut est donc :

$$
a = (a_1,\dots,a_p).
$$

### 13.5 Normalisation euclidienne

Le vecteur de poids final est :

$$
w =
\frac{a}{\|a\|_2}
$$

si $\|a\|_2 > \varepsilon$.

Sinon, le code prend un vecteur uniforme :

$$
w_j = \frac{1}{\sqrt{p}}, \qquad j=1,\dots,p.
$$

### 13.6 Operateur de score

Le score de la fenetre $i$ est :

$$
s_i = \sum_{j=1}^p Z_{ij} w_j.
$$

En notation matricielle :

$$
s = Zw.
$$

### 13.7 Correlation d'entrainement

Le fichier sauve aussi :

$$
\rho_{\mathrm{train}} = \rho(s, t).
$$

Cette quantite mesure l'alignement empirique entre le score appris et la cible `gain_effective` sur les fenetres qui ont servi a apprendre le score.

### 13.8 Interpretation precise

Ce point est crucial :

La fonction de score n'est pas une regression lineaire au sens moindres carres.

Elle est un operateur lineaire dont les poids sont construits a partir de correlations marginales standardisees.

Cela signifie :

- un poids positif indique qu'une feature est positivement associee a `gain_effective`
- un poids negatif indique l'inverse
- les effets conditionnels entre variables ne sont pas modelises
- les correlations entre features peuvent influencer l'interpretation des poids

En bref :

$$
w_j \approx \text{importance marginale standardisee de la feature } j
$$

et non coefficient causal ou coefficient OLS conditionnel.

### 13.9 Application du score appris

Pour appliquer le score a de nouvelles fenetres, le code impose :

- meme ordre de colonnes
- meme liste de features retenues
- meme standardisation $(\mu_j,\sigma_j)$

Si $X_{\mathrm{new}}$ est aligne sur les bonnes features :

$$
Z_{\mathrm{new},ij}
=
\frac{X_{\mathrm{new},ij}-\mu_j}{\sigma_j},
$$

puis :

$$
s_{\mathrm{new}} = Z_{\mathrm{new}} w.
$$

## 14. Objectif optimise par Optuna

Le script `optuna_xgb_simple.py` recharge la fonction de score apprise, puis cherche des hyperparametres XGBoost qui maximisent ce score.

### 14.1 Hyperparametres optimises

Optuna explore un espace $\Theta$ portant sur :

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `reg_lambda`

### 14.2 Protocole d'evaluation pour un jeu de parametres

Pour un $\theta \in \Theta$, le code :

1. recharge le dataset `digits`
2. reconstruit la cible binaire $y_i = \mathbf{1}[z_i \ge 5]$
3. effectue un melange aleatoire global des indices
4. rejoue la boucle sequentielle par 170 fenetres de 10 observations
5. calcule les metriques de chaque fenetre
6. applique la fonction de score sauvegardee sur les features `score_model["feature_names"]`

On obtient ainsi 170 scores de fenetre :

$$
s_1(\theta),\dots,s_{170}(\theta).
$$

### 14.3 Objectif Optuna

L'objectif maximise est :

$$
J(\theta) = \mathrm{median}\left(s_1(\theta),\dots,s_{170}(\theta)\right).
$$

Donc Optuna cherche :

$$
\theta^\star = \arg\max_{\theta \in \Theta} J(\theta).
$$

### 14.4 Point de vigilance important

`main.py` peut apprendre le score sur des donnees "stochastic" produites par `temporal_dataset.py`.

En revanche, `optuna_xgb_simple.py` n'utilise pas `temporal_dataset.py` :

- il repart directement du dataset `digits`
- puis melange les indices

Il existe donc potentiellement un decalage de distribution entre :

- le regime de generation du score
- le regime d'optimisation par Optuna

Mathematiquement, cela signifie que la fonction objectif optimise par Optuna n'est pas necessairement evaluee sur la meme loi empirique que celle ayant servi a construire le score.

## 15. Scores structurels du module `structural_scores.py`

Ce module n'est pas central dans `main.py`, mais il formalise plusieurs scores composites.

### 15.1 Brique de base : moyenne geometrique ponderee

Pour des valeurs non negatives $x_1,\dots,x_K$ et des poids positifs $a_1,\dots,a_K$ :

$$
\mathrm{WGM}(x;a)
=
\prod_{k=1}^K x_k^{a_k / \sum_{u=1}^K a_u}.
$$

Le code verifie :

- meme forme entre valeurs et poids
- somme des poids strictement positive
- toutes les valeurs non negatives

### 15.2 Fonction de clipping

$$
\mathrm{clip01}(v) = \min(\max(v,0),1).
$$

### 15.3 Score de separation

Le module definit :

$$
\mathrm{auc\_norm}(a) = \mathrm{clip01}\left(\frac{a-0.5}{0.5}\right),
$$

$$
\mathrm{threshold\_confidence\_ratio\_norm}(r)
=
\mathrm{clip01}\left(\frac{r-1.0}{1.5}\right),
$$

$$
\mathrm{threshold\_confidence\_gap\_norm}(g)
=
\mathrm{clip01}\left(\frac{g}{0.12}\right).
$$

Puis :

$$
\mathrm{margin\_quality}(r,g)
=
\mathrm{WGM}\left(
\left[
\mathrm{threshold\_confidence\_ratio\_norm}(r),
\mathrm{threshold\_confidence\_gap\_norm}(g)
\right];
[0.55,0.45]
\right).
$$

Enfin :

$$
\mathrm{separation\_raw\_score}(a,r,g)
=
\mathrm{WGM}\left(
\left[
\mathrm{auc\_norm}(a),
\mathrm{margin\_quality}(r,g)
\right];
[0.75,0.25]
\right).
$$

### 15.4 Score de calibration

Le baseline de Brier vaut :

$$
\mathrm{brier\_baseline}(\pi) = \pi(1-\pi).
$$

Le gain relatif de log-loss vaut :

$$
\mathrm{relative\_logloss\_gain}(\ell,\ell_0)
=
\frac{\ell_0-\ell}{\ell_0}.
$$

Puis :

$$
\mathrm{logloss\_gain\_norm}(\ell,\ell_0)
=
\mathrm{clip01}\left(
\frac{\mathrm{relative\_logloss\_gain}(\ell,\ell_0)}{0.20}
\right),
$$

$$
\mathrm{logloss\_abs\_norm}(\ell)
=
\mathrm{clip01}\left(\frac{0.69-\ell}{0.69-0.45}\right),
$$

$$
\mathrm{logloss\_norm}(\ell,\ell_0)
=
\mathrm{WGM}\left(
\left[
\mathrm{logloss\_gain\_norm}(\ell,\ell_0),
\mathrm{logloss\_abs\_norm}(\ell)
\right];
[0.50,0.50]
\right).
$$

Pour le Brier :

$$
\mathrm{brier\_gain\_norm}(b,b_0)
=
\mathrm{clip01}\left(\frac{(b_0-b)/b_0}{0.25}\right).
$$

Le score brut de calibration vaut :

$$
\mathrm{calibration\_raw\_score}(\ell,\ell_0,b,\pi)
=
\mathrm{WGM}\left(
\left[
\mathrm{logloss\_norm}(\ell,\ell_0),
\mathrm{brier\_gain\_norm}(b,\pi(1-\pi))
\right];
[0.60,0.40]
\right).
$$

### 15.5 Score de decision

Le module definit d'abord :

$$
\mathrm{signal\_quality}(p,r,s)
=
\mathrm{WGM}([p,r,s];[0.40,0.35,0.25]).
$$

Puis :

$$
\mathrm{threshold\_alignment}(\tau,\tau_{\mathrm{impl}})
=
\mathrm{clip01}(1-|\tau-\tau_{\mathrm{impl}}|),
$$

$$
\mathrm{positive\_rate\_alignment}(\mathrm{ppr},\pi)
=
\mathrm{clip01}(1-|\mathrm{ppr}-\pi|).
$$

Ensuite :

$$
\mathrm{threshold\_coherence}
=
\mathrm{WGM}\left(
\left[
\mathrm{threshold\_alignment},
\mathrm{positive\_rate\_alignment}
\right];
[0.55,0.45]
\right).
$$

Enfin :

$$
\mathrm{decision\_raw\_score}
=
\mathrm{WGM}\left(
\left[
\mathrm{signal\_quality},
\mathrm{threshold\_coherence}
\right];
[0.70,0.30]
\right).
$$

### 15.6 Score de generalisation

Le module propose :

$$
\mathrm{overfit\_norm}(o) = \mathrm{clip01}\left(1-\frac{o}{0.20}\right),
$$

$$
\mathrm{variance\_norm}(v) = \mathrm{clip01}\left(1-\frac{v}{0.03}\right),
$$

$$
\mathrm{auc\_stability\_norm}(s) = \mathrm{clip01}\left(1-\frac{s}{0.20}\right),
$$

$$
\mathrm{bias\_norm}(b) = \mathrm{clip01}\left(\frac{0.85-b}{0.85-0.55}\right).
$$

Puis :

$$
\mathrm{generalization\_raw\_score}
=
\mathrm{WGM}\left(
\left[
\mathrm{overfit\_norm},
\mathrm{variance\_norm},
\mathrm{auc\_stability\_norm},
\mathrm{bias\_norm}
\right];
[0.35,0.25,0.20,0.20]
\right).
$$

## 16. Metriques de generalisation du module `generalization_metrics.py`

Ce module propose des estimateurs simples de robustesse inter-fold.

### 16.1 Biais empirique

Si $L^{\mathrm{val}}_1,\dots,L^{\mathrm{val}}_K$ sont les losses de validation :

$$
\mathrm{empirical\_bias} =
\frac{1}{K}\sum_{k=1}^K L^{\mathrm{val}}_k.
$$

### 16.2 Variance empirique

$$
\mathrm{empirical\_variance}
=
\mathrm{Var}(L^{\mathrm{val}}_1,\dots,L^{\mathrm{val}}_K).
$$

### 16.3 Bruit residuel empirique

Pour des verites terrain $y_i$ et des scores $s_i$ :

$$
\mathrm{empirical\_residual\_noise}
=
\mathrm{Var}(y_i-s_i).
$$

### 16.4 Overfitting empirique

Si $L^{\mathrm{train}}_k$ et $L^{\mathrm{val}}_k$ sont les losses train et validation :

$$
\mathrm{empirical\_overfitting}
=
\frac{1}{K}\sum_{k=1}^K L^{\mathrm{val}}_k
-
\frac{1}{K}\sum_{k=1}^K L^{\mathrm{train}}_k.
$$

### 16.5 Stabilite

Pour une suite de valeurs $v_1,\dots,v_K$ :

$$
\mathrm{stability\_score}
=
\frac{1}{1+\mathrm{std}(v_1,\dots,v_K)}.
$$

### 16.6 Pire fold

Si "higher is better" :

$$
\mathrm{worst\_fold} = \min_k v_k.
$$

Sinon :

$$
\mathrm{worst\_fold} = \max_k v_k.
$$

### 16.7 Ecart de robustesse

Si "higher is better" :

$$
\mathrm{robustness\_gap}
=
\frac{1}{K}\sum_{k=1}^K v_k - \min_k v_k.
$$

Sinon :

$$
\mathrm{robustness\_gap}
=
\max_k v_k - \frac{1}{K}\sum_{k=1}^K v_k.
$$

### 16.8 Ecarts-types inter-fold

Le module fournit aussi :

- `recall_std`
- `precision_std`
- `ppr_std`
- `logloss_std`
- `auc_std`
- `brier_std`
- `sharpe_std`
- `threshold_std`
- `interfold_std`

qui ne sont rien d'autre que :

$$
\mathrm{std}(v_1,\dots,v_K).
$$

Il fournit egalement :

- `interfold_variance` : variance inter-fold
- `threshold_drift = \max(v_k) - \min(v_k)`
- `worst_sharpe = \min(\text{Sharpe}_k)`
- `max_drawdown_worst = \max(\text{Drawdown}_k)`

## 17. Conventions numeriques et cas degeneres

Pour bien interpreter les resultats, il faut garder en tete plusieurs conventions d'implementation.

### 17.1 Divisions par zero

Dans la plupart des modules metriques :

$$
\frac{a}{0} \mapsto 0.
$$

Ce choix simplifie la stabilite numerique, mais il impose une convention semantique forte.

### 17.2 Masques vides

Quand une moyenne est prise sur un ensemble vide, le code renvoie :

$$
0.0.
$$

Cela concerne notamment les distances conditionnelles (`dist_tp`, `dist_fp`, etc.).

### 17.3 AUC d'une fenetre monoclasse

Si une fenetre ne contient que des positifs ou que des negatifs :

$$
\mathrm{AUC} = \mathrm{NaN}
$$

dans le dictionnaire de metriques.

Ensuite :

- pour les correlations paire par paire, ces `NaN` sont ignores si possible
- pour les calculs de rang, ils sont remplaces par `0.0`
- pour l'apprentissage du score, ils sont ignores dans le calcul de $\mu_j$ et $\sigma_j$, puis remplaces par `0.0` apres standardisation

### 17.4 Nature du score appris

Le score appris est :

- lineaire
- standardise
- base sur des correlations marginales

Ce n'est pas :

- un estimateur bayesien
- une regression OLS
- une fonction de cout directement derivee de la theorie de la decision

## 18. Lecture conceptuelle d'ensemble

Le projet peut etre resume mathematiquement comme suit.

On construit, pour chaque fenetre $t$, un vecteur de metriques ex ante :

$$
m_t \in \mathbb{R}^{17},
$$

et une cible de performance economique :

$$
t_t = \mathrm{gain\_effective}^{(t)}.
$$

On cherche ensuite un operateur lineaire :

$$
S(m_t) = w^\top \, \mathrm{standardize}(m_t)
$$

tel que :

$$
S(m_t)
$$

soit fortement correle a :

$$
t_t.
$$

Autrement dit, le projet cherche une reponse a la question suivante :

> Quelles proprietes mesurables d'une fenetre de classification permettent d'anticiper au mieux sa qualite economique ?

La fonction de score apprise est la reponse empirique produite par le code.
