from enum import Enum


class ModelEnum(Enum):
    BAYESIAN = "bayesian"
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGB_CLASSIFIER = "xgb_classifier"
