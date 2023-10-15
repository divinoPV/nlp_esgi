from sklearn.ensemble import RandomForestClassifier
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


def make_model(
    feature_extraction: FeatureExtractionEnum,
    model: ModelEnum,
):
    return Pipeline(
        [
            (
                feature_extraction.value,
                {
                    FeatureExtractionEnum.COUNT_VECTORIZER: CountVectorizer(),
                    FeatureExtractionEnum.HASHING_VECTORIZER: HashingVectorizer(n_features=1000),
                    FeatureExtractionEnum.TFIDF_VECTORIZER: TfidfVectorizer(),
                    FeatureExtractionEnum.TFIDF_VECTORIZER_W_NGRAM: TfidfVectorizer(ngram_range=(1, 2)),
                }[feature_extraction],
            ),
            (
                model.value,
                {
                    ModelEnum.BAYESIAN: MultinomialNB,
                    ModelEnum.LOGISTIC_REGRESSION: LogisticRegression,
                    ModelEnum.RANDOM_FOREST: RandomForestClassifier,
                    ModelEnum.XGB_CLASSIFIER: XGBClassifier,
                }[model](),
            ),
        ]
    )
