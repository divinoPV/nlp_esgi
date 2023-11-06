from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.enum.feature_extraction import FeatureExtractionEnum
from src.enum.model import ModelEnum
from src.features.make_features import preprocess_text, stopwords


def make_model(
    feature_extraction: FeatureExtractionEnum,
    model: ModelEnum,
    task: str,
):
    if task == "is_name":
        feature_extraction_value = FeatureExtractionEnum.DICT_VECTORIZER.value
        feature_extraction_object = DictVectorizer(sparse=False)
    elif task == "is_comic_video":
        feature_extraction_value = feature_extraction.value
        feature_extraction_object = {
            FeatureExtractionEnum.COUNT_VECTORIZER: CountVectorizer(stop_words=stopwords, preprocessor=preprocess_text),
            FeatureExtractionEnum.HASHING_VECTORIZER: HashingVectorizer(n_features=1_000, stop_words=stopwords, preprocessor=preprocess_text),
            FeatureExtractionEnum.TFIDF_VECTORIZER: TfidfVectorizer(stop_words=stopwords, preprocessor=preprocess_text),
        }[feature_extraction]

    model = (
        model.value,
        {
            ModelEnum.BAYESIAN: MultinomialNB(alpha=.9),
            ModelEnum.LOGISTIC_REGRESSION: LogisticRegression(max_iter=1_000),
            ModelEnum.RANDOM_FOREST: RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, max_features=.9),
            ModelEnum.XGB_CLASSIFIER: XGBClassifier(),
        }[model],
    )

    if task == "find_comic_name":
        return Pipeline([model])
    else:
        return Pipeline([(feature_extraction_value, feature_extraction_object), model])
