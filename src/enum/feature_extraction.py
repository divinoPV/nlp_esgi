from enum import Enum


class FeatureExtractionEnum(Enum):
    COUNT_VECTORIZER = "count_vectorizer"
    HASHING_VECTORIZER = "hashing_vectorizer"
    TFIDF_VECTORIZER = "tfidf_vectorizer"
    TFIDF_VECTORIZER_W_NGRAM = "tfidf_vectorizer"
