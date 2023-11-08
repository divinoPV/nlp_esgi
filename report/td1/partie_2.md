# Partie 2 : Évaluation des modèles pour la tâche "is_name"

Dans cette seconde partie, nous avons mené des expérimentations pour évaluer les performances de divers modèles de classification dans le contexte de la prédiction de noms de personnes. Nous avons également analysé les résultats pour déterminer quel modèle est le plus adapté à cette tâche.

## Évaluation des modèles de classification
Nous avons testé plusieurs modèles de classification.

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
Accuracy : 46.56%
F1-Score : 46.01%

### LOGISTIC_REGRESSION
Accuracy : 46.02%
F1-Score : 45.38%

### BAYESIAN
Accuracy : 46.04%
F1-Score : 44.53%

## Conclusion
En conclusion, les résultats montrent que tous les modèles évalués, à savoir Random Forest, Logistic Regression et Bayesian, ont des performances relativement similaires avec des scores d'Accuracy et de F1-Score proches de 46%. Cependant, ces performances restent modestes, indiquant qu'il peut être nécessaire d'explorer d'autres modèles ou techniques de traitement des données pour améliorer la capacité de classification dans cette tâche particulière. Il peut également être utile d'examiner de plus près les caractéristiques des données d'entraînement et d'ajuster les paramètres des modèles pour voir s'il est possible d'obtenir de meilleures performances.

Bien que la différence de performances entre les modèles ne soit pas très significative, il est important de noter que d'autres facteurs tels que la taille de l'ensemble de données et la complexité du modèle peuvent également influencer le choix final du modèle.

## À suivre
Dans la prochaine section du TD, nous adapterons le code pour prédire si une vidéo est humoristique et, le cas échéant, pour extraire le nom du comique. Nous testerons l'ensemble des modèles que nous avons évalués jusqu'à présent afin de déterminer le modèle le plus optimal pour cette tâche.
