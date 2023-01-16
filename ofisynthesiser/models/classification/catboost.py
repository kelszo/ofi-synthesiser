import numpy as np
from catboost import CatBoostClassifier
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score


def hyper_opt_catboost(
    Model: CatBoostClassifier,
    X: np.ndarray,
    y: np.ndarray,
    timeout: int,
    seed: int,
    **kwargs,
) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    def Objective(trial: optuna.Trial):
        param = {
            "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
            "eval_metric": trial.suggest_categorical("eval_metric", ["AUC"]),
            "random_state": trial.suggest_categorical("random_state", [seed]),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 5.5, step=0.5),
            "verbose": trial.suggest_categorical("verbose", [0]),
            "train_dir": "./tmp",
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

        model = Model(**param)
        score = cross_val_score(model, X, y, scoring="roc_auc", cv=skf).mean()

        return score

    study = optuna.create_study(
        direction="maximize",
        study_name="CatBoost optimization",
    )
    study.optimize(Objective, gc_after_trial=True, timeout=timeout)

    model = Model(random_state=seed, verbose=0)
    base_score = cross_val_score(model, X, y, scoring="roc_auc", cv=skf).mean()

    hyperopt_score = study.best_value
    best_params = study.best_params

    if base_score > hyperopt_score:
        best_params = {}

    best_params["verbose"] = 0
    best_params["random_state"] = seed
    best_params["train_dir"] = "./tmp"

    model = Model(**best_params)
    model.fit(X, y)

    return best_params
