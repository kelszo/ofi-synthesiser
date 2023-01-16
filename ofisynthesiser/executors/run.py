import pickle
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from ofisynthesiser.data import preprocessor, target
from ofisynthesiser.utils import logging
from ofisynthesiser.models.classification import (
    SVC,
    CatBoostClassifier,
    ExtraTreesClassifier,
    KNeighborsClassifier,
    LGBMClassifier,
    LogisticRegression,
    RandomForestClassifier,
    XGBClassifier,
    hyper_opt_model,
)
from ofisynthesiser.models.synthesising import generate_data_copula_gan, generate_data_tvae
from ofisynthesiser.utils import seed_everything

start_time = datetime.now().strftime("%y%m%d-%H%M")

SEED = 2023
RESAMPLES = 100
HYPEROPT_TIMEOUT = 60 * 60
DEBUG = True
HYPERPARAMS_PATH = ""

if DEBUG:
    logging.info("DEBUGGING MODE")
    RESAMPLES = 3
    HYPEROPT_TIMEOUT = 1

seed_everything(SEED)

data = pd.read_csv("./data/raw/combined_dataset.csv")
# Remove Na OFI cases
data = data[data[target].notna()]
data[target] = data[target].replace("No", 0)
data[target] = data[target].replace("Yes", 1)

best_params = {}
save_params = False

if HYPERPARAMS_PATH != "":
    with open(HYPERPARAMS_PATH, "rb") as handle:
        best_params = pickle.load(handle)

classification_models = {
    "CATBOOST": CatBoostClassifier,
    "EXTRATREES": ExtraTreesClassifier,
    "KNN": KNeighborsClassifier,
    "LIGHTGBM": LGBMClassifier,
    "LOGIT": LogisticRegression,
    "RANDOMFOREST": RandomForestClassifier,
    "SVM": SVC,
    "XGBOOST": XGBClassifier,
}

model_names = classification_models.keys()

synthesising_models = {"NONE": None}
synth_method_names = synthesising_models.keys()

results = []

for resample_idx in range(RESAMPLES):
    resample_results = {
        model_name: {synth_method_name: [] for synth_method_name in synth_method_names} for model_name in model_names
    }

    logging.info(f"RESAMPLE:{resample_idx}")
    df_raw, df_test = train_test_split(data, test_size=0.2, random_state=resample_idx)

    y_raw = df_raw[target].to_numpy()
    X_raw = df_raw.drop(columns=[target])
    X_raw = preprocessor.fit_transform(X_raw)

    y_test = df_test[target].to_numpy()
    X_test = df_test.drop(columns=[target])
    X_test = preprocessor.transform(X_test)

    resample_results["TARGET"] = y_test

    for model_name in classification_models:
        Model = classification_models[model_name]

        if resample_idx == 0 and model_name not in best_params:
            logging.info(f"HYPER OPTIMISING: {model_name}")
            best_params[model_name] = hyper_opt_model(Model, X_raw, y_raw, timeout=HYPEROPT_TIMEOUT, seed=SEED)
            save_params = True

        model = Model(**best_params[model_name])

        model.fit(X_raw, y_raw)
        resample_results[model_name]["NONE"] = model.predict_proba(X_test)[:, 1]

    for synth_name in synthesising_models:
        synth = synthesising_models[synth_name]

        if synth is None:
            continue

        df_synth = synth(df_raw, debug=DEBUG)

        y_synth = df_synth[target].to_numpy()
        X_synth = df_synth.drop(columns=[target])
        X_synth = preprocessor.transform(X_synth)

        y_comb = y_raw + y_synth
        X_comb = X_raw + X_synth

        for model_name in classification_models:
            Model = classification_models[model_name]
            model = Model(**best_params[model_name])

            model.fit(X_synth, y_synth)
            resample_results[model_name][synth_name] = model.predict_proba(X_test)[:, 1]

            model.fit(X_comb, y_comb)
            resample_results[model_name][f"{synth_name}:COMB"] = model.predict_proba(X_test)[:, 1]

    results.append(resample_results)

if save_params:
    with open(f"out/{start_time}-hyperparams.pickle", "wb") as handle:
        pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f"out/{start_time}-results.pickle", "wb") as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
