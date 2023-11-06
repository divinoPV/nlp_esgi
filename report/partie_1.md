# Partie 1 : Classification de texte, valuation des modèles pour la tâche "is_comic_video"

Dans cette première partie, nous avons exploré les caractéristiques potentielles et les modèles pour la tâche de classification des vidéos humoristiques. Nous avons également examiné l'impact individuel de différentes variations dans le choix des caractéristiques et des modèles, et nous avons tiré des conclusions sur les modèles appropriés pour cette tâche.

## Préconceptions sur les caractéristiques et les modèles
Nous avions des hypothèses sur les caractéristiques et les modèles qui pourraient être utiles pour cette tâche.

### Textuelles
Nous nous attendions à ce que les caractéristiques textuelles, telles que TFIDF_VECTORIZER, COUNT_VECTORIZER, HASHING_VECTORIZER, soient pertinentes pour cette tâche. Nous pensions que ces caractéristiques pourraient capturer des informations importantes dans les titres des vidéos, mais les résultats obtenus étaient trop faibles pour une prédiction fiable.

### Modèles
Nous supposions que des modèles de classification traditionnels comme la LOGISTIC_REGRESSION et le RANDOM_FOREST, le BAYESIAN et le XGB_CLASSIFIER seraient adéquats pour cette tâche. En effet, RANDOM_FOREST et LOGISTIC_REGRESSION sont généralement efficaces pour les classifications de ce type, mais on pourrait penser que  XGB_CLASSIFIER et XGB_CLASSIFIER pourraient offrir de meilleures performances dans cette tâche.

## Influence des variations
Au cours de notre travail, nous avons examiné différentes variations dans les caractéristiques et les modèles, et nous avons observé leur impact individuel :

### Textuelles
TFIDF_VECTORIZER vs COUNT_VECTORIZER vs HASHING_VECTORIZER : Nous avons constaté que le modèle utilisant COUNT_VECTORIZER offrait généralement de meilleures performances que le modèle utilisant le TFIDF_VECTORIZER ou HASHING_VECTORIZER. COUNT_VECTORIZER permet diviser le texte en mots, ou "tokens", en utilisant un délimiteur. Il compte le nombre d'occurrences de chaque mot/token du vocabulaire dans les documents.

### Modèles
LOGISTIC_REGRESSION vs RANDOM_FOREST vs XGB_CLASSIFIER vs BAYESIAN : Nous avons constaté que le RANDOM_FOREST surpassait généralement la LOGISTIC_REGRESSION, BAYESIAN et XGB_CLASSIFIER en termes de performances. Le RANDOM_FOREST est un modèle plus complexe capable de mieux saisir les relations non linéaires dans les données, ce qui est avantageux pour cette tâche.

## Résultats
### RANDOM_FOREST - COUNT_VECTORIZER
Accuracy : 92.15%
F1-Score : 86.74%

### RANDOM_FOREST - HASHING_VECTORIZER
Accuracy : 90.52%
F1-Score : 83.79%

### RANDOM_FOREST - TFIDF_VECTORIZER
Accuracy : 91.77%
F1-Score : 85.54%

### LOGISTIC_REGRESSION - COUNT_VECTORIZER
Accuracy : 90.65%
F1-Score : 81.60%

### LOGISTIC_REGRESSION - HASHING_VECTORIZER
Accuracy : 85.65%
F1-Score : 66.13%

### LOGISTIC_REGRESSION - TFIDF_VECTORIZER
Accuracy : 85.03%
F1-Score : 64.68%

### BAYESIAN - COUNT_VECTORIZER
Accuracy : 90.89%
F1-Score : 83.88%

### BAYESIAN - TFIDF_VECTORIZER
Accuracy : 84.27%
F1-Score : 60.71%

### XGB_CLASSIFIER - COUNT_VECTORIZER
Accuracy : 91.02%
F1-Score : 84.38%

### XGB_CLASSIFIER - HASHING_VECTORIZER
Accuracy : 89.27%
F1-Score : 80.91%

### XGB_CLASSIFIER - TFIDF_VECTORIZER
Accuracy : 90.27%
F1-Score : 81.85%

## Conclusions sur le modèle optimal
En conclusion, nos résultats montrent que les modèles Random Forest et XGBoost ont globalement donné de bonnes performances, en particulier lorsqu'ils ont été utilisés avec le Count Vectorizer. Les régressions logistiques ont également donné de bons résultats, mais les modèles bayésiens avec TF-IDF Vectorizer ont montré des performances inférieures. Il est essentiel de choisir le modèle et le vectoriseur qui conviennent le mieux à notre tâche de classification en fonction des métriques de performance que nous privilégions, qu'il s'agisse de l'Accuracy, du F1-Score ou d'autres métriques spécifiques à notre domaine d'application.

## À suivre
Dans la prochaine partie de l'exercice dirigé, nous adapterons le code pour permettre l'entraînement du modèle afin d'extraire les noms des comiques réalisant les chroniques.
