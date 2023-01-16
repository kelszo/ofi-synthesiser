from typing import Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import optuna
from sklearn.metrics import roc_auc_score


class KNN(KNeighborsClassifier):
    def __init__(self, seed: int, **kwargs):
        self.seed = seed
        super(KNN, self).__init__(**kwargs)

    def hyper_opt(self, X, y, eval_set: Tuple[np.ndarray, np.ndarray], timeout: int) -> dict:
        X_valid, y_valid = eval_set

        def Objective(trial: optuna.Trial):
            param = {
                "n_neighbors": trial.suggest_int("n_neighbors", [3, 10]),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "leaf_size": trial.suggest_int("leaf_size", [10, 100]),
            }

            self.load_from_params(param)

            self.fit(X, y)
            probas_valid = self.predict_proba(X_valid)

            score = roc_auc_score(y_valid, probas_valid)

            return score

        study = optuna.create_study(
            direction="maximize",
            study_name="k-NN optimization",
        )
        study.optimize(Objective, gc_after_trial=True, timeout=timeout)

        best_params = study.best_params

        # retrain
        self.load_from_params(best_params)
        self.fit(X, y)

        return best_params
