from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

from custom_enum.feature_extraction import FeatureExtractionEnum
from custom_enum.model import ModelEnum
from features.make_features import preprocess_text, stopwords


def make_model(
    feature_extraction: FeatureExtractionEnum,
    model: ModelEnum,
    task: str,
):
    model = {
        ModelEnum.BAYESIAN: MultinomialNB(alpha=.9),
        ModelEnum.LOGISTIC_REGRESSION: LogisticRegression(max_iter=1_000, random_state=42),
        ModelEnum.RANDOM_FOREST: RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, max_features=.9),
        ModelEnum.XGB_CLASSIFIER: XGBClassifier(),
    }[model]

    if task == "is_comic_video":
        return make_pipeline(
            {
                FeatureExtractionEnum.COUNT_VECTORIZER: CountVectorizer(stop_words=stopwords, preprocessor=preprocess_text),
                FeatureExtractionEnum.HASHING_VECTORIZER: HashingVectorizer(n_features=1_000, stop_words=stopwords, preprocessor=preprocess_text),
                FeatureExtractionEnum.TFIDF_VECTORIZER: TfidfVectorizer(stop_words=stopwords, preprocessor=preprocess_text),
            }[feature_extraction],
            SMOTE(random_state=42),
            model,
        )

    if task == "is_name":
        return make_pipeline(
            ColumnTransformer(
                [
                    ("tfidf_vectorizer_1", TfidfVectorizer(preprocessor=preprocess_text), "previous_word"),
                    ("tfidf_vectorizer_2", TfidfVectorizer(preprocessor=preprocess_text), "following_word"),
                ],
            ),
            RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, max_features=.9),
        )

    if task == "find_comic_name":
        return make_pipeline(model)
