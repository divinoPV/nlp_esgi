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



