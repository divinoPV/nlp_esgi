# Rapport sur le Projet de NLP

## A-priori sur les features et modèles

Au début du projet, j'avais plusieurs a-priori sur les features et les modèles qui seraient utiles pour résoudre les tâches de classification de texte et de reconnaissance d'entités nommées. Ces a-priori ont influencé mes choix initiaux pour le développement du modèle. Voici un aperçu de mes réflexions :

### Features

1. **Count Vectorizer**: Je pensais que la représentation basée sur le compte des mots pourrait être utile pour la classification de texte comique, car les comiques ont tendance à utiliser des mots spécifiques ou récurrents dans leurs chroniques. De plus, la simplicité de cette méthode la rendait intéressante pour une première approche.

2. **Hashing Vectorizer**: J'ai envisagé l'utilisation du Hashing Vectorizer pour gérer l'explosion combinatoire de fonctionnalités. Cependant, j'étais conscient que cela pourrait entraîner une perte d'informations, mais je souhaitais expérimenter pour évaluer son impact.

3. **TF-IDF Vectorizer**: Le TF-IDF est une technique couramment utilisée en NLP, et je pensais qu'elle pourrait aider à identifier les mots clés importants dans le texte, ce qui pourrait être précieux pour la classification des vidéos humoristiques.

4. **TF-IDF Vectorizer avec N-grammes**: J'ai également envisagé l'utilisation de N-grammes pour capturer des informations de contexte plus complexes. Je pensais que cela pourrait améliorer la performance du modèle, en particulier pour la tâche de reconnaissance des noms de personnes.

### Modèles

1. **Naïve Bayes (MultinomialNB)**: Le modèle bayésien naïf est souvent utilisé pour la classification de texte en raison de sa simplicité et de ses performances acceptables. Je l'ai considéré comme une option de base pour la classification de texte comique.

2. **Régression logistique (Logistic Regression)**: La régression logistique est un autre modèle couramment utilisé en NLP. Je pensais qu'elle serait utile pour la classification de texte et la reconnaissance des entités nommées.

3. **Random Forest Classifier**: J'ai choisi le RandomForest pour sa capacité à gérer des ensembles de données complexes et son potentiel pour capturer des relations non linéaires entre les caractéristiques.

4. **XGBoost Classifier**: Bien que XGBoost ne provienne pas de scikit-learn, je l'ai inclus car il est largement utilisé en compétition de machine learning et est connu pour sa puissance de prédiction. Je pensais qu'il pourrait donner de très bons résultats.

## Apports individuels de chaque variation

Pour évaluer les performances des features et des modèles, j'ai effectué plusieurs expérimentations en combinant différentes options. Voici les apports individuels de chaque variation que j'ai testée :

### Features

1. **Count Vectorizer**: Cette méthode s'est avérée efficace pour la classification de texte comique. Elle a permis de capturer des mots spécifiques aux comiques. Cependant, pour la reconnaissance des noms de personnes, elle n'a pas donné des résultats satisfaisants, car elle ne tient pas compte du contexte.

2. **Hashing Vectorizer**: Malheureusement, le Hashing Vectorizer a produit des résultats moins performants que le Count Vectorizer. La perte d'informations due à la réduction de dimensionnalité a eu un impact négatif.

3. **TF-IDF Vectorizer**: Le TF-IDF Vectorizer s'est avéré être une amélioration par rapport au Count Vectorizer pour la reconnaissance des noms de personnes. Il a aidé à identifier les mots clés importants.

4. **TF-IDF Vectorizer avec N-grammes**: L'ajout de N-grammes a considérablement amélioré la performance du modèle pour la tâche de reconnaissance des noms de personnes. Il a permis de capturer des relations de contexte plus complexes entre les mots.

### Modèles

1. **Naïve Bayes (MultinomialNB)**: MultinomialNB a donné des performances décentes pour la classification de texte comique, mais il a montré ses limites pour la reconnaissance des noms de personnes en raison de sa simplicité.

2. **Régression logistique (Logistic Regression)**: La régression logistique a été une option solide pour les deux tâches. Elle a montré une bonne capacité à apprendre des relations entre les caractéristiques et les étiquettes.

3. **Random Forest Classifier**: Le modèle Random Forest s'est avéré efficace pour la classification de texte comique, mais il a montré des performances légèrement inférieures à la régression logistique pour la reconnaissance des noms de personnes.

4. **XGBoost Classifier**: XGBoost a généralement donné les meilleurs résultats, en particulier pour la classification de texte comique. Cependant, il est plus complexe à paramétrer et nécessite plus de temps de calcul.

## Conclusion sur le bon modeling

Après avoir effectué de nombreuses expérimentations, je peux tirer quelques conclusions sur le bon modeling dans l'état actuel du projet :

1. Pour la classification de texte comique, la combinaison d'un TF-IDF Vectorizer avec un modèle XGBoost Classifier s'est avérée la plus performante. Cela a permis de capturer les mots clés importants et de modéliser des relations non linéaires.

2. Pour la reconnaissance des noms de personnes, un TF-IDF Vectorizer avec des N-grammes associé à une régression logistique a donné des résultats solides. Cela a permis de prendre en compte le contexte et d'identifier les noms de personnes.

3. La régression logistique s'est révélée être un choix fiable pour les deux tâches, offrant un bon équilibre entre simplicité et performance.

4. Il est essentiel de garder une trace écrite des expérimentations, car cela permet de comprendre pourquoi certaines variations ont réussi ou échoué et d'orienter les choix futurs.

En résumé, le bon modeling dépend de la tâche spécifique à accomplir. Il est crucial d'expérimenter avec différentes combinaisons de features et de modèles pour trouver la meilleure approche. De plus, l'utilisation de XGBoost s'est avérée particulièrement puissante pour la classification de texte comique. Cependant, la régression logistique est une alternative solide et plus facile à paramétrer.