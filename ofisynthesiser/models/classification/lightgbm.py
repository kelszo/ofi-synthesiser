from typing import Tuple

import numpy as np
from lightgbm import LGBMClassifier
import optuna
from sklearn.metrics import roc_auc_score


class LightGBM(LGBMClassifier):
    def __init__(self, seed: int, random_state=0, **kwargs):
        self.seed = seed
        random_state = seed
        self.random_state = seed
        super(LightGBM, self).__init__(**kwargs, random_state=seed)

    def hyper_opt(self, X, y, eval_set: Tuple[np.ndarray, np.ndarray], timeout: int) -> dict:
        X_valid, y_valid = eval_set

        def Objective(trial: optuna.Trial):
            param = {
                "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
                "depth": trial.suggest_int("depth", 1, 12),
                "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
                "eval_metric": trial.suggest_categorical("eval_metric", ["AUC"]),
                "random_state": trial.suggest_categorical("random_state", [self.seed]),
                "cat_features": self.get_cat_feature_indices(),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 5.5, step=0.5),
                "train_dir": "./tmp",
            }

            if param["objective"] == "Logloss":
                _, counts = np.unique(y, return_counts=True)
                scale_pos_weight = counts[0] / counts[1]
                param["scale_pos_weight"] = trial.suggest_categorical("scale_pos_weight", [scale_pos_weight])

            if param["bootstrap_type"] == "Bayesian":
                param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
            elif param["bootstrap_type"] == "Bernoulli":
                param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

            self.load_from_params(param)

            self.fit(X, y, eval_set=[(X_valid, y_valid)])

            probas_valid = self.predict_proba(X_valid)

            score = roc_auc_score(y_valid, probas_valid)

            return score

        study = optuna.create_study(
            direction="maximize",
            study_name="CatBoost optimization",
        )
        study.optimize(Objective, gc_after_trial=True, timeout=timeout)

        best_params = study.best_params
        best_params["cat_features"] = self.get_cat_feature_indices()
        best_params["train_dir"] = "./tmp"

        # retrain
        self.load_from_params(best_params)
        self.fit(X, y)

        return best_params
