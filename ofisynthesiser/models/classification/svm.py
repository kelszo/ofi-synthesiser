from typing import Tuple

import numpy as np
from sklearn.svm import SVC
import optuna
from sklearn.metrics import roc_auc_score


class SVM(SVC):
    def __init__(self, seed: int, **kwargs):
        self.seed = seed
        super(SVM, self).__init__(**kwargs, class_weight="balanced", probability=True)

    def hyper_opt(self, X, y, eval_set: Tuple[np.ndarray, np.ndarray], timeout: int) -> dict:
        X_valid, y_valid = eval_set

        def Objective(trial: optuna.Trial):
            param = {
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
                "kernel": trial.suggest_categorical("loss", ["linear", "poly", "rbf", "sigmoid"]),
                "C": trial.suggest_loguniform("C", 0.1, 1000),
            }

            self.load_from_params(param)

            self.fit(X, y)
            probas_valid = self.predict_proba(X_valid)

            score = roc_auc_score(y_valid, probas_valid)

            return score

        study = optuna.create_study(
            direction="maximize",
            study_name="SVM optimization",
        )
        study.optimize(Objective, gc_after_trial=True, timeout=timeout)

        best_params = study.best_params

        # retrain
        self.load_from_params(best_params)
        self.fit(X, y)

        return best_params
