# NLP TD

L'objectif de ce TD est de créer un modèle "nom de vidéo" -> "nom du comique" si c'est la chronique d'un comique, None sinon. On formulera le problème en 2 tâches d'apprentissage:
- une de text classification pour savoir si la vidéo est une chronique comique
- une de named-entity recognition pour reconnaître les noms dans un texte
En assemblant les deux, nous obtiendrons notre modèle.

Dans ce TD, on s'intéresse surtout à la démarche. Pour chaque tâche:
- Bien poser le problème
- Avoir une baseline
- Experimenter diverses features et modèles
- Garder une trace écrite des expérimentations dans un rapport. Dans le rapport, on s'intéresse plus au sens du travail effectué (quelles expérimentations ont été faites, pourquoi, quelles conclusions) qu'à la liste de chiffres.
- Avoir une codebase clean, permettant de reproduire les expérimentations.

On se contentera de méthodes pré-réseaux de neurones. Nos features sont explicables et calculables "à la main".

La codebase doit fournir les entry points suivant:
- Un entry point pour train sur une "task", prenant en entrée le path aux données de train et dumpant le modèle dans "model_dump" 
```
python src/main.py train --task=is_comic_video --input_file=data/raw/train.csv --model_dump=models/model.json
```
- Un entry point pour predict sur une "task", prenant en entrée le path au modèle dumpé, le path aux données à prédire et outputtant dans un csv les prédictions
```
python src/main.py predict --task=is_comic_video --input_file=data/raw/test.csv --model_dump=models/model.json --output_file=data/processed/prediction.csv
```
- Un entry point pour evaluer un modèle sur une "task", prenant en entrée le path aux données de train.
```
python src/main.py evaluate --task=is_comic_video --input_file=data/raw/train.csv
```

Les "tasks":
- "is_comic_video": prédit si la video est une chronique comique
- "is_name": prédit si le mot est un nom de personne
- "find_comic_name": si la video est une chronique comique, sort le nom du comique

## Partie 1: Text classification: prédire si la vidéo est une chronique comique

### Librairies

- sklearn.feature_extraction.text: Regarder les features disponibles. Lesquelles semblent adaptées ?
- nltk: télécharger le corpus français. La librairie permettra de retirer les stopwords et de stemmer les mots



