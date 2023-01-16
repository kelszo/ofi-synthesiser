import pdb

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from ofisynthesiser.data import target, preprocessor
from ofisynthesiser.models.classification import CatBoost, XGBoost, LightGBM, SVM, KNN, Logit, ExtraTrees, RandomForest
from ofisynthesiser.models.synthesising import CTGAN

from ofisynthesiser.utils import seed_everything

SEED = 2023
RESAMPLES = 100
DEBUG = True

if DEBUG:
    print("DEBUGGING MODE")
    RESAMPLES = 3

seed_everything(SEED)

data = pd.read_csv("./data/raw/combined_dataset.csv")
# Remove Na OFI cases
data = data[data[target].notna()]
data[target] = data[target].replace("No", 0)
data[target] = data[target].replace("Yes", 1)

classification_models = {
    "CATBOOST": CatBoost,
    "XGBOOST": XGBoost,
    "LIGHTGBM": LightGBM,
    "SVM": SVM,
    "KNN": KNN,
    "LOGIT": Logit,
    "EXTRATREES": ExtraTrees,
    "RANDOMFOREST": RandomForest,
}

synthesising_models = {"NONE": None}

for resample_idx in range(RESAMPLES):
    print(f"RESAMPLE:{resample_idx}")
    df_raw, df_test = train_test_split(data, test_size=0.2, random_state=resample_idx)

    for synth_name in synthesising_models:
        Synth = synthesising_models[synth_name]
        y_raw = df_raw[target].to_numpy()
        X_raw = df_raw.drop(columns=[target])
        X_raw = preprocessor.fit_transform(X_raw)

        y_test = df_test[target].to_numpy()
        X_test = df_test.drop(columns=[target])
        X_test = preprocessor.transform(X_test)

        if Synth is not None:
            synth = Synth(seed=SEED, debug=DEBUG)
            synth.fit(df_raw)

            df_synth = synth.sample(len(df_raw))

            y_synth = df_synth[target].to_numpy()
            X_synth = df_synth.drop(columns=[target])
            X_synth = preprocessor.transform(X_synth)

            y_comb = y_raw + y_synth
            X_comb = X_raw + X_synth

        for model_name in classification_models:
            Model = classification_models[model_name]

            model = Model(seed=SEED)

            model.fit(X_raw, y_raw)
            probas_raw = model.predict_proba(X_test)[:, 1]
            score_raw = roc_auc_score(y_test, probas_raw)

            if Synth is not None:
                model.fit(X_synth, y_synth)
                probas_synth = model.predict_proba(X_test)[:, 1]
                score_synth = roc_auc_score(y_test, probas_synth)

                model.fit(X_comb, y_comb)
                probas_comb = model.predict_proba(X_test)[:, 1]
                score_comb = roc_auc_score(y_test, probas_comb)

            if Synth is not None:
                print("MODEL:", model_name)
                print("RAW SCORE:", score_raw)
                print("SYNTH SCORE:", score_synth)
                print("COMB SCORE:", score_comb, end="\n")
            else:
                print("MODEL:", model_name)
                print("RAW SCORE:", score_raw, end="\n")
