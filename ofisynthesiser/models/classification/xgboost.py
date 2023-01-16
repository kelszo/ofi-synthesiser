from typing import Tuple, List
import numpy as np
from xgboost import XGBClassifier
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def hyper_opt_xgboost(
    Model: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    timeout: int,
    seed: int,
    **kwargs,
) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    def Objective(trial: optuna.Trial):
        param = {
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
            "random_state": trial.suggest_categorical("random_state", [seed]),
            "eval_metric": trial.suggest_categorical("eval_metric", ["auc"]),
        }

        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
            param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
            param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

        model = Model(**param)
        score = cross_val_score(model, X, y, scoring="roc_auc", cv=skf).mean()

        return score

    study = optuna.create_study(
        direction="maximize",
        study_name="XGBoost optimization",
    )
    study.optimize(Objective, gc_after_trial=True, timeout=timeout)

    model = Model()
    base_score = cross_val_score(model, X, y, scoring="roc_auc", cv=skf).mean()

    hyperopt_score = study.best_value
    best_params = study.best_params

    if base_score > hyperopt_score:
        best_params = {}

    best_params["random_state"] = seed

    model = Model(**best_params)
    model.fit(X, y)

    return best_params
