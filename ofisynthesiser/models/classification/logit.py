from typing import Tuple, List
import numpy as np
from sklearn.linear_model import LogisticRegression
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def hyper_opt_logit(
    Model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    timeout: int,
    seed: int,
    **kwargs,
) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    def Objective(trial: optuna.Trial):
        param = {
            "C": trial.suggest_float("logreg_c", 1e-10, 1e10),
            "max_iter": trial.suggest_categorical("max_iter", [2500]),
        }

        model = Model(**param)
        score = cross_val_score(model, X, y, scoring="roc_auc", cv=skf).mean()

        return score

    study = optuna.create_study(
        direction="maximize",
        study_name="Logit optimization",
    )
    study.optimize(Objective, gc_after_trial=True, timeout=timeout)

    model = Model()
    base_score = cross_val_score(model, X, y, scoring="roc_auc", cv=skf).mean()

    hyperopt_score = study.best_value
    best_params = study.best_params

    if base_score > hyperopt_score:
        best_params = {}

    model = Model(**best_params)
    model.fit(X, y)

    return best_params
