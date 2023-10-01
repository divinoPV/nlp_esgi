from src.data.make_dataset import make_dataset
from src.features.make_features import make_features

class DumbModel:
    def fit(self, X, y):
        pass

    def predict(self, X):
        return [0] * len(X)

    def dump(self, filename_output):
        pass

def train_model(input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df)

    model = DumbModel()
    model.fit(X, y)

    return model.dump(model_dump_filename)

