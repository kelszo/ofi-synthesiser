import pickle
import statistics
import numpy as np
from pprint import pprint

from sklearn.metrics import roc_auc_score
import scipy.stats as st

results = None

with open("out/230116-2003-results.pickle", "rb") as handle:
    results = pickle.load(handle)

model_names = list(results[0].keys())
model_names.remove("TARGET")
synth_method_names = list(results[0][model_names[0]].keys())

auc_per_resample = {model_name: {synth_method: [] for synth_method in synth_method_names} for model_name in model_names}

for resample in results:
    target = resample["TARGET"]
    del resample["TARGET"]

    for model_name in resample:
        model_results = resample[model_name]
        for synth_method in model_results:
            probas = model_results[synth_method]
            auc_score = roc_auc_score(target, probas)

            auc_per_resample[model_name][synth_method].append(auc_score)

for model_name in auc_per_resample:
    print("-----------")
    print(f"{model_name}:")
    model_results = auc_per_resample[model_name]
    for synth_method in model_results:
        aucs = model_results[synth_method]

        median = np.round(statistics.median(aucs), 3)
        ci = np.round(st.norm.interval(alpha=0.95, loc=np.mean(aucs), scale=st.sem(aucs)), 3)

        print(f"{synth_method}: {median} ({ci[0]}, {ci[1]})")
    print("-----------\n")