from typing import Tuple, List
import numpy as np
from sklearn.svm import SVC
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def hyper_opt_svm(
    Model: SVC,
    X: np.ndarray,
    y: np.ndarray,
    timeout: int,
    seed: int,
    **kwargs,
) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    def Objective(trial: optuna.Trial):
        param = {
            # "kernel": trial.suggest_categorical("loss", ["linear", "poly", "rbf", "sigmoid"]),
            "C": trial.suggest_loguniform("C", 0.1, 1000),
            "random_state": trial.suggest_categorical("random_state", [seed]),
            "probability": trial.suggest_categorical("probability", [True]),
        }

        model = Model(**param)
        score = cross_val_score(model, X, y, scoring="roc_auc", cv=skf).mean()

        return score

    study = optuna.create_study(
        direction="maximize",
        study_name="SVM optimization",
    )
    study.optimize(Objective, gc_after_trial=True, timeout=timeout)

    model = Model(probability=True)
    base_score = cross_val_score(model, X, y, scoring="roc_auc", cv=skf).mean()

    hyperopt_score = study.best_value
    best_params = study.best_params

    if base_score > hyperopt_score:
        best_params = {}

    best_params["random_state"] = seed
    best_params["probability"] = True

    model = Model(**best_params)
    model.fit(X, y)

    return best_params
