import click
import joblib
import numpy as np
import warnings

from pandas import DataFrame
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

from src.data.make_dataset import make_dataset
from src.enum.feature_extraction import FeatureExtractionEnum
from src.enum.model import ModelEnum
from src.features.make_features import make_features
from src.model.main import make_model


# Ignorer les avertissements
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost.data")
warnings.filterwarnings("ignore", category=UserWarning, message="The least populated class in y has only 1 members, which is less than n_splits=5.")


@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--feature_extraction", default="COUNT_VECTORIZER", help="Type of feature extraction")
@click.option("--model_type", default="LOGISTIC_REGRESSION", help="Type of model")
def train(
    task: str,
    input_filename: str,
    model_dump_filename: str,
    feature_extraction: str,
    model_type: str,
):
    if "is_comic_video" != task and ModelEnum[model_type.strip()] == ModelEnum.XGB_CLASSIFIER:
        raise Exception("You cannot choose an XGBoost for the is_name and find_comic_name tasks.")

    df = make_dataset(input_filename)
    X, y = make_features(df, task)

    model = make_model(
        feature_extraction=FeatureExtractionEnum[feature_extraction.strip()],
        model=ModelEnum[model_type.strip()],
        task=task,
    )
    model.fit(X, y)

    joblib.dump(model, model_dump_filename)

    print(f"Model saved to {model_dump_filename}")


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/test.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def predict(
    task: str,
    input_filename: str,
    model_dump_filename: str,
    output_filename: str,
):
    df = make_dataset(input_filename)
    X, _ = make_features(df, task)  # You only need X for prediction

    model = joblib.load(model_dump_filename)
    predictions = model.predict(X)

    # Save predictions to a CSV
    output_df = DataFrame({'predictions': predictions})
    output_df.to_csv(output_filename, index=False)

    print(f"Predictions saved to {output_filename}")


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--feature_extraction", default="COUNT_VECTORIZER", help="Type of feature extraction")
@click.option("--model_type", default="LOGISTIC_REGRESSION", help="Type of model")
def evaluate(
    task: str,
    input_filename: str,
    feature_extraction: str,
    model_type: str,
):
    if "is_comic_video" != task and ModelEnum[model_type.strip()] == ModelEnum.XGB_CLASSIFIER:
        raise Exception("You cannot choose an XGBoost for the is_name and find_comic_name tasks.")

    df = make_dataset(input_filename)
    X, y = make_features(df, task)

    model = make_model(
        feature_extraction=FeatureExtractionEnum[feature_extraction.strip()],
        model=ModelEnum[model_type.strip()],
        task=task,
    )

    # Scikit learn has function for cross validation
    accuracy_scores = cross_val_score(model, X, y, scoring="accuracy")
    f1_scores = cross_val_score(model, X, y, scoring="f1_macro")

    print(f"Mean accuracy: {100 * np.mean(accuracy_scores)}%")
    print(f"Mean F1 Score: {100 * np.mean(f1_scores)}%")

    return accuracy_scores, f1_scores


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
