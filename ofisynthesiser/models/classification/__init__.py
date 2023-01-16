from ofisynthesiser.models.classification.catboost import hyper_opt_catboost, CatBoostClassifier
from ofisynthesiser.models.classification.extratrees import hyper_opt_extra_trees, ExtraTreesClassifier
from ofisynthesiser.models.classification.knn import hyper_opt_knn, KNeighborsClassifier
from ofisynthesiser.models.classification.lightgbm import hyper_opt_lightgbm, LGBMClassifier
from ofisynthesiser.models.classification.logit import hyper_opt_logit, LogisticRegression
from ofisynthesiser.models.classification.randomforest import hyper_opt_random_forest, RandomForestClassifier
from ofisynthesiser.models.classification.svm import hyper_opt_svm, SVC
from ofisynthesiser.models.classification.xgboost import hyper_opt_xgboost, XGBClassifier

import numpy as np


def hyper_opt_model(Model: any, X: np.ndarray, y: np.ndarray, timeout: int, seed: int, **kwargs) -> any:
    if Model is CatBoostClassifier:
        return hyper_opt_catboost(Model, X, y, timeout, seed, **kwargs)
    elif Model is ExtraTreesClassifier:
        return hyper_opt_extra_trees(Model, X, y, timeout, seed, **kwargs)
    elif Model is KNeighborsClassifier:
        return hyper_opt_knn(Model, X, y, timeout, seed, **kwargs)
    elif Model is LGBMClassifier:
        return hyper_opt_lightgbm(Model, X, y, timeout, seed, **kwargs)
    elif Model is LogisticRegression:
        return hyper_opt_logit(Model, X, y, timeout, seed, **kwargs)
    elif Model is RandomForestClassifier:
        return hyper_opt_random_forest(Model, X, y, timeout, seed, **kwargs)
    elif Model is SVC:
        return hyper_opt_svm(Model, X, y, timeout, seed, **kwargs)
    elif Model is XGBClassifier:
        return hyper_opt_xgboost(Model, X, y, timeout, seed, **kwargs)
