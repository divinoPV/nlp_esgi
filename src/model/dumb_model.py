from json import dump

class DumbModel:
    """Dumb model always predict 0"""
    def fit(self):
        pass

    def predict(self, X):
        return [0] * len(X)

    def dump(self, filename_output):
        with open(filename_output, "w+") as file:
            try:
                dump([], file)
                print(f"Fichier {filename_output} mis à jour !")
            except Exception as e:
                print(e)
