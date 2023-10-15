import click
import numpy as np
import warnings

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.model_selection import cross_val_score

from src.data.make_dataset import make_dataset
from src.enum.feature_extraction import FeatureExtractionEnum
from src.enum.model import ModelEnum
from src.features.make_features import make_features
from src.model.main import make_model


# Ignorer les avertissements relatifs à is_sparse
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost.data")
stemmer = SnowballStemmer("french")


@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df, task)

    model = make_model()
    model.fit(X, y)

    return model.dump(model_dump_filename)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def test(task, input_filename, model_dump_filename, output_filename):
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--feature_extraction", default="COUNT_VECTORIZER", help="Type of feature extraction")
@click.option("--model_type", default="LINEAR_REGRESSION", help="Type of model")
def evaluate(
    task: str,
    input_filename: str,
    feature_extraction: str,
    model_type: str,
):
    df = make_dataset(input_filename)
    X, y = make_features(df, task)

    if "is_comic_video" != task and ModelEnum[model_type.strip()] == ModelEnum.XGB_CLASSIFIER:
        raise Exception("You cannot choose an XGBoost for the is_name and find_comic_name tasks.")

    model = make_model(
        feature_extraction=FeatureExtractionEnum[feature_extraction.strip()],
        model=ModelEnum[model_type.strip()],
    )

    # Scikit learn has function for cross validation
    scores = cross_val_score(model, X, y, scoring="accuracy")

    print(f"Got accuracy {100 * np.mean(scores)}%")

    return scores


def preprocess_text(text):
    # Tokenization du texte
    words = word_tokenize(text, language='french')
    
    # Retirer les stopwords
    french_stopwords = set(stopwords.words('french'))
    filtered_words = [word for word in words if word.lower() not in french_stopwords]
    
    # Stemming des mots
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    
    # Rejoindre les mots en une seule chaîne
    processed_text = ' '.join(stemmed_words)
    
    return processed_text


cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
