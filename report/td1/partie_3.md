# Partie 3 : Évaluation des modèles pour la tâche "find_comic_name"

Dans cette dernière partie du TD, nous avons évalué les performances des différents modèles de classification pour la tâche "find_comic_name". Nous avons analysé les résultats obtenus pour chaque modèle, y compris l'accuracy et le F1-score.

## Évaluation des modèles de classification
Nous avons testé plusieurs modèles de classification, à savoir :

### RANDOM_FOREST
Nous avons utilisé le modèle RANDOM_FOREST en raison de sa capacité à capturer des relations complexes dans les données. Il s'agit d'un modèle d'ensemble robuste qui combine plusieurs arbres de décision.

### LOGISTIC_REGRESSION
La LOGISTIC_REGRESSION est un modèle linéaire couramment utilisé pour la classification. Nous l'avons inclus en tant que modèle de base pour évaluer ses performances.

### BAYESIAN
Un modèle BAYESIAN est un modèle statistique qui repose sur le théorème de Bayes pour estimer des probabilités et prendre en compte des informations antérieures dans la prise de décision. Il utilise une approche probabiliste pour modéliser l'incertitude et la variabilité des données, en combinant des connaissances a priori avec des données observées pour obtenir des estimations plus précises.

## Hyperparamètres
### RANDOM_FOREST
- n_estimators : Le nombre d'arbres de décision dans la forêt, réglé à 200 dans cet exemple.
- max_depth : La profondeur maximale des arbres, non spécifiée dans cet exemple.
- random_state : La graine aléatoire utilisée pour l'initialisation de la forêt, fixée à 42.
- max_features : Le nombre maximal de caractéristiques à considérer lors de la recherche de la meilleure partition à chaque nœud de l'arbre, fixé à 0.9 (90 % des caractéristiques sont utilisées).

### LOGISTIC_REGRESSION
- max_iter : Le nombre maximal d'itérations pour la convergence de l'optimisation, réglé à 1000.

### BAYESIAN
- alpha : Contrôle la force de la régularisation (ou la pénalisation) appliquée aux coefficients du modèle, réglé à 0.9.

## Résultats
### RANDOM_FOREST
Accuracy : 87.55%
F1-Score : 87.49%

### LOGISTIC_REGRESSION
Accuracy : 82.52%
F1-Score : 82.44%

### BAYESIAN
Accuracy : 81.34%
F1-Score : 81.24%

## Conclusion
En conclusion, les résultats démontrent que le modèle Random Forest a obtenu les meilleures performances avec une accuracy de 87.55% et un F1-Score de 87.49%, suivi de près par la Logistic Regression avec une accuracy de 82.52% et un F1-Score de 82.44%. Le modèle Bayesian a également montré des performances solides, bien que légèrement inférieures aux deux autres modèles, avec une accuracy de 81.34% et un F1-Score de 81.24%.

Dans l'ensemble, ces performances sont encourageantes et indiquent une capacité de classification efficace des modèles sur la tâche donnée. Il serait judicieux d'explorer davantage les données et de considérer d'autres métriques ou techniques pour optimiser les modèles, mais dans l'ensemble, ces résultats sont positifs.

Cependant, il est important de noter que le choix du modèle dépend également des contraintes en termes de temps d'entraînement et de ressources.

Il est recommandé de continuer à affiner les modèles, d'explorer d'autres hyperparamètres et de prendre en considération les spécificités de l'ensemble de données pour prendre des décisions finales sur le modèle à utiliser.
