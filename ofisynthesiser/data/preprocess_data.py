from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, OrdinalEncoder

continuous_features = [
    "dt_alarm_hosp",
    "dt_alarm_scene",
    "dt_ed_emerg_proc",
    "dt_ed_first_ct",
    "dt_ed_norm_be",
    "ed_be_art",
    "ed_inr",
    "ed_rr_value",
    "ed_sbp_value",
    "hosp_los_days",
    "hosp_vent_days",
    "ISS",
    "iva_dagar_n",
    "iva_vardtillfallen_n",
    "NumberOfActions",
    "NumberOfInjuries",
    "pre_rr_value",
    "pre_sbp_value",
    "pt_age_yrs",
]

ordinal_features = [
    "ed_rr_rtscat",
    "ed_sbp_rtscat",
    "ed_gcs_motor",
    "ed_gcs_sum",
    "pre_gcs_motor",
    "pre_gcs_sum",
    "res_gos_dischg",
    "pt_asa_preinjury",
    "pre_rr_rtscat",
    "pre_sbp_rtscat",
    "pre_provided",
]

nominal_features = [
    "AlarmRePrioritised",
    "ed_be_art_NotDone",
    "ed_emerg_proc_other",
    "ed_emerg_proc",
    "ed_inr_NotDone",
    "ed_intub_type",
    "ed_intubated",
    "ed_tta",
    "FirstTraumaDT_NotDone",
    "hosp_dischg_dest",
    "host_care_level",
    "host_transfered",
    "host_vent_days_NotDone",
    "inj_dominant",
    "inj_intention",
    "inj_mechanism",
    "pre_card_arrest",
    "pre_intub_type",
    "pre_intubated",
    "pre_transport",
    "pt_Gender",
    "res_survival",
    "TraumaAlarmAtHospital",
    "TraumaAlarmCriteria",
]

features = continuous_features + ordinal_features + nominal_features
discrete_features = ordinal_features + nominal_features

target = "ofi"


class SweTrauFix(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X["ed_inr"] = pd.to_numeric(X["ed_inr"].str.replace(",", "."))
        X["ed_be_art"] = pd.to_numeric(X["ed_be_art"].str.replace(",", "."))

        X[discrete_features] = X[discrete_features].replace(99, np.nan)
        X[discrete_features] = X[discrete_features].replace(999, np.nan)
        X[discrete_features] = X[discrete_features].replace(9999, np.nan)

        rr_rtscats = {"ed_rr_rtscat": "ed_rr_value", "pre_rr_rtscat": "pre_rr_value"}
        sbp_rtscats = {"ed_sbp_rtscat": "ed_sbp_value", "pre_sbp_rtscat": "pre_sbp_value"}

        rr_rts_values = {4: 15, 3: 40, 2: 7, 1: 3, 0: 0}
        sbp_rts_values = {4: 100, 3: 83, 2: 62, 1: 25, 0: 0}

        for rr_rtscat_feature in rr_rtscats:
            value_feat = rr_rtscats[rr_rtscat_feature]
            for rr_rtscat in rr_rts_values:
                average_value = rr_rts_values[rr_rtscat]
                X.loc[(X[rr_rtscat_feature] == rr_rtscat) & (X[value_feat].isna()), value_feat] = average_value

        for sbp_rtscat_feature in sbp_rtscats:
            value_feat = sbp_rtscats[sbp_rtscat_feature]
            for sbp_rtscat in sbp_rts_values:
                average_value = sbp_rts_values[sbp_rtscat]
                X.loc[(X[sbp_rtscat_feature] == sbp_rtscat) & (X[value_feat].isna()), value_feat] = average_value

        return X


continuous_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", PowerTransformer()),
    ]
)

ordinal_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # ("ordinalencoder", OrdinalEncoder(handle_unknown="error")),
    ]
)

nominal_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("onehotencoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

feature_processor = ColumnTransformer(
    transformers=[
        ("continuous", continuous_transformer, continuous_features),
        ("ordinal", ordinal_transformer, ordinal_features),
        ("nominal", nominal_transformer, nominal_features),
    ]
)

preprocessor = Pipeline(
    steps=[
        ("swetraufix", SweTrauFix()),
        ("featureprocessor", feature_processor),
    ]
)
# data[discrete_features] = data[discrete_features].astype(discrete_dtype)
# data[target] = data[target].astype(discrete_dtype)


def get_feat_types(data: pd.DataFrame) -> List[str]:
    feat_types = []
    for column in data.columns:
        feat_type = None
        for categorical_feature in nominal_features:
            if column.startswith(categorical_feature):
                feat_type = "Categorical"
                break

        if feat_type is None:
            feat_type = "Numerical"

        feat_types.append(feat_type)


def get_category_idxs(data: pd.DataFrame) -> Tuple[List[int], List[int]]:
    features = data.columns.to_list()
    feat_types = get_feat_types(data)

    cat_idxs = []
    cat_dims = []

    for i, (feature, feat_type) in enumerate(zip(features, feat_types)):
        if feat_type == "Categorical":
            cat_idxs.append(i)
            cat_dims.append(len(data[feature].unique()))

    return cat_idxs, cat_dims


def get_features(data: pd.DataFrame) -> Tuple[List[str], List[str]]:
    features = data.columns.to_list()
    feat_types = get_feat_types(data)

    categorical = []
    continuous = []

    for feature, feat_type in zip(features, feat_types):
        if feat_type == "Categorical":
            categorical.append(feature)
        else:
            continuous.append(feature)

    return categorical, continuous
