from tableone import TableOne
import pandas as pd
import numpy as np

from report.figures.format_tableone import format_tableone

from ofisynthesiser.data.preprocess_data import SweTrauFix, continuous_features, ordinal_features, nominal_features


def table1(data):
    intubated_columns = ["pre_intubated", "ed_intubated"]
    data[intubated_columns] = data[intubated_columns].replace({1: True, 2: False})
    data["intubated"] = data["pre_intubated"] | data["ed_intubated"]
    data["intubated"] = data["intubated"].replace({True: "Yes", False: "No"})

    data["pt_Gender"] = data["pt_Gender"].replace({1: "Male", 2: "Female"})

    data["res_survival"] = data["res_survival"].replace({1: "Yes", 2: "No"})

    data["host_care_level"] = data["host_care_level"].replace(
        {1: "ED", 2: "General ward", 3: "Surgical ward", 4: "Specialist/Intermediate ward", 5: "Intensive care unit"}
    )

    data["ed_emerg_proc"] = data["ed_emerg_proc"].replace(
        {
            1: "Thoracotomy",
            2: "Laparotomy",
            3: "Pelvis packing",
            4: "Revascularisation",
            5: "Radiological intervention",
            6: "Craniotomy",
            7: "Intracranial pressure measurement",
            8: "Other",
        }
    )

    mask = data[["ed_emerg_proc"]].applymap(lambda x: isinstance(x, (str)))

    data[["ed_emerg_proc"]] = data[["ed_emerg_proc"]].where(mask)

    data["ofi"] = data["ofi"].replace({"Yes": "OFI", "No": "No OFI"})

    columns = [
        "pt_age_yrs",
        "pt_Gender",
        "res_survival",
        "host_care_level",
        "ISS",
        "ed_rr_value",
        "ed_sbp_value",
        "ed_gcs_sum",
        "dt_ed_first_ct",
        "dt_ed_emerg_proc",
        "intubated",
        "ed_emerg_proc",
        "ofi",
    ]

    categorical = ["pt_Gender", "res_survival", "host_care_level", "intubated", "ed_emerg_proc", "ofi"]

    rename = {
        "pt_age_yrs": "Age",
        "pt_Gender": "Gender",
        "res_survival": "Dead at 30 days",
        "host_care_level": "Highest level of care",
        "ISS": "Injury severity score",
        "ed_rr_value": "ED Respiratory Rate",
        "ed_gcs_sum": "ED GCS",
        "ed_sbp_value": "ED Systolic blood pressure",
        "dt_ed_first_ct": "Time to first CT",
        "dt_ed_emerg_proc": "Time to definitive treatment",
        "intubated": "Intubated",
        "ed_emerg_proc": "Emergency procedure",
    }

    reverse_rename = {v: k for k, v in rename.items()}

    data = data[columns]
    data[categorical] = data[categorical].astype("category")

    data = data.rename(columns=rename)

    table1 = format_tableone(data, "ofi")

    print(table1.to_latex(index=False, escape=False))


baseline_data = pd.read_csv("data/raw/combined_dataset.csv")
baseline_data = baseline_data[baseline_data["ofi"].notna()]
baseline_data = SweTrauFix().transform(baseline_data)
table1(baseline_data)

ctgan_data = pd.read_csv("data/raw/ctgan_data.csv")
table1(ctgan_data)

tvae_data = pd.read_csv("data/raw/tvae_data.csv")
table1(tvae_data)
