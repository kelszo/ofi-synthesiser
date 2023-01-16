from typing import Tuple, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def hyper_opt_random_forest(
    Model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    timeout: int,
    seed: int,
    **kwargs,
) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    def Objective(trial: optuna.Trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 1, 150),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 60),
            "random_state": trial.suggest_categorical("random_state", [seed]),
        }

        model = Model(**param)
        score = cross_val_score(model, X, y, scoring="roc_auc", cv=skf).mean()

        return score

    study = optuna.create_study(
        direction="maximize",
        study_name="RandomForest optimization",
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
