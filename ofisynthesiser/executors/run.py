import os
import pickle
from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from ofisynthesiser.data import preprocessor, target
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
from ofisynthesiser.utils import logging, seed_everything, tqdm_joblib

start_time = datetime.now().strftime("%y%m%d-%H%M")

SEED = 2023
N_RESAMPLES = 100
HYPEROPT_TIMEOUT = 60 * 60
DEBUG = False
HYPERPARAMS_PATH = ""

seed_everything(SEED)

if torch.cuda.is_available():
    logging.info(f"SUCCESSFULLY USING CUDA")
    CUDA = True
else:
    logging.info(f"FAILED TO LOAD CUDA")
    CUDA = False
    DEBUG = True

if DEBUG:
    logging.info("DEBUGGING MODE")
    N_RESAMPLES = 3
    HYPEROPT_TIMEOUT = 1

data = pd.read_csv("./data/raw/combined_dataset.csv")
# Remove Na OFI cases
data = data[data[target].notna()]
data[target] = data[target].replace("No", 0)
data[target] = data[target].replace("Yes", 1)

best_params = {}

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

synthesising_models = {"NONE": None, "CTGAN": generate_data_copula_gan, "TVAE": generate_data_tvae}

results = []

with open(f"out/{start_time}.lock", "w") as f:
    f.write("")

# First resample split
df_raw, df_test = train_test_split(data, test_size=0.2, random_state=0)
y_raw = df_raw[target].to_numpy()
X_raw = df_raw.drop(columns=[target])
X_raw = preprocessor.fit_transform(X_raw)


def hyperopt(Model, model_name, X, y):
    logging.info(f"HYPER OPTIMISING: {model_name}")

    params = hyper_opt_model(Model, X, y, timeout=HYPEROPT_TIMEOUT, seed=SEED)

    return {model_name: params}


if HYPERPARAMS_PATH != "":
    with open(HYPERPARAMS_PATH, "rb") as handle:
        best_params = pickle.load(handle)
else:
    with tqdm_joblib(tqdm(total=len(classification_models))) as progress_bar:
        collected_best_params = Parallel(n_jobs=os.cpu_count())(
            delayed(hyperopt)(classification_models[model_name], model_name, X_raw, y_raw)
            for model_name in classification_models
        )

    # Merge params
    best_params = {k: v for d in collected_best_params for k, v in d.items()}

    with open(f"out/{start_time}-hyperparams.pickle", "wb") as handle:
        pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


def resample(
    data: pd.DataFrame,
    resample_idx: int,
):
    try:
        resample_results = {
            model_name: {synth_method_name: [] for synth_method_name in synthesising_models.keys()}
            for model_name in classification_models.keys()
        }

        resample_results["RESAMPLE"] = resample_idx

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
            model = Model(**best_params[model_name])

            model.fit(X_raw, y_raw)
            resample_results[model_name]["NONE"] = model.predict_proba(X_test)[:, 1]

        for synth_name in synthesising_models:
            synth = synthesising_models[synth_name]

            if synth is None:
                continue

            df_synth = synth(df_raw, cuda=CUDA, debug=DEBUG)

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

        return resample_results
    except Exception as e:
        logging.error(e)
        return {}


with tqdm_joblib(tqdm(total=N_RESAMPLES)) as progress_bar:
    results = Parallel(n_jobs=os.cpu_count())(
        delayed(resample)(data, resample_idx) for resample_idx in range(N_RESAMPLES)
    )


with open(f"out/{start_time}-results.pickle", "wb") as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
