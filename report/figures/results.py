import pickle
import statistics
import numpy as np
import pandas as pd

from sklearn import metrics
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import style

format_classification_models = {
    "CATBOOST": "CatBoost",
    "EXTRATREES": "Extra Trees",
    "KNN": "k-NN",
    "LIGHTGBM": "LightGBM",
    "LOGIT": "Logistic Regression",
    "RANDOMFOREST": "Random forest",
    "SVM": "SVM",
    "XGBOOST": "XGBoost",
}

format_synth_models = {
    "NONE": "Baseline",
    "CTGAN": "CTGAN",
    "TVAE": "TVAE",
    "CTGAN:COMB": "CTGAN + Baseline",
    "TVAE:COMB": "TVAE + Baseline",
}

results = None

with open("out/230209-1412-results.pickle", "rb") as f:
    results = pickle.load(f)

model_names = list(results[0].keys())
model_names.remove("TARGET")
model_names.remove("RESAMPLE")
synth_method_names = list(results[0][model_names[0]].keys())

auc_per_resample = {model_name: {synth_method: [] for synth_method in synth_method_names} for model_name in model_names}

for resample in results:
    target = resample["TARGET"]
    resample_idx = resample["RESAMPLE"]
    del resample["TARGET"]
    del resample["RESAMPLE"]

    for model_name in resample:
        model_results = resample[model_name]
        for synth_method in model_results:
            probas = model_results[synth_method]
            auc_score = metrics.roc_auc_score(target, probas)

            auc_per_resample[model_name][synth_method].append(auc_score)

main_result_table = pd.DataFrame(index=model_names, columns=synth_method_names)

best_data_source = {model_name: [0, ""] for model_name in model_names}

for model_name in auc_per_resample:
    model_results = auc_per_resample[model_name]
    for synth_method in model_results:
        aucs = model_results[synth_method]

        median = np.round(statistics.median(aucs), 3)
        ci = np.round(st.norm.interval(alpha=0.95, loc=np.mean(aucs), scale=st.sem(aucs)), 3)

        main_result_table.loc[model_name, synth_method] = f"{median}\\newline({ci[0]}, {ci[1]})"

        if median > best_data_source[model_name][0]:
            best_data_source[model_name] = [median, synth_method]

for model_name in best_data_source:
    _, data_source = best_data_source[model_name]

    main_result_table.loc[model_name, data_source] = "\\textbf{" + main_result_table.loc[model_name, data_source] + "}"

main_result_table.index = map(lambda x: format_classification_models[x], main_result_table.index)
main_result_table.columns = map(lambda x: format_synth_models[x], main_result_table.columns)

print(main_result_table.to_latex(escape=False))

with open("out/230209-1412-results.pickle", "rb") as f:
    results = pickle.load(f)

plt.rcParams.update(
    {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Serif"],
    }
)

plt.figure()

unpacked = {synth_model: [] for synth_model in synth_method_names}
unpacked["TARGET"] = []

for resample in results:
    for model_name in model_names:
        for synth_method in synth_method_names:
            unpacked[synth_method].extend(resample[model_name][synth_method])

        unpacked["TARGET"].extend(resample["TARGET"])

for synth_method in synth_method_names:
    probas = unpacked[synth_method]
    target = unpacked["TARGET"]

    fpr, tpr, _ = metrics.roc_curve(target, probas)
    auc = metrics.roc_auc_score(target, probas)

    plt.plot(fpr, tpr, label=f"{format_synth_models[synth_method]} (AUC = {auc:.{3}f})")

plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="grey", label="No discrimination")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(title="Synthesising model", loc="lower right", fontsize=9)
plt.style.use("seaborn-paper")
plt.savefig("report/figures/roc.pdf")
