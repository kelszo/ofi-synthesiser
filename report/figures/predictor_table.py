import pandas as pd

format_features = {
    "dt_alarm_hosp": "DT alarm to ED",
    "dt_alarm_scene": "DT alarm to arrival at scene",
    "dt_ed_emerg_proc": "DT arrival at ED to emergency procedure",
    "dt_ed_first_ct": "DT arrival at ED to first CT",
    "dt_ed_norm_be": "DT arrival ED to normalised base excess",
    "ed_be_art": "First base excess at the ED",
    "ed_inr": "First Prothrombin time (international normalized ratio)",
    "ed_rr_value": "Respiratory rate at ED",
    "ed_sbp_value": "Systolic blood pressure at ED",
    "hosp_los_days": "Total admittance days at hospital",
    "hosp_vent_days": "Total amount of days in a ventilator",
    "ISS": "Injury Severity Score",
    "iva_dagar_n": "Total amount of days in the intensive care unit",
    "iva_vardtillfallen_n": "Amount of times admitted to the intensive care unit during admittance",
    "NumberOfActions": "Number of Actions done",
    "NumberOfInjuries": "Number of injuries",
    "pre_rr_value": "PH Respiratory rate",
    "pre_sbp_value": "PH Systolic blood pressure",
    "pt_age_yrs": "Patient age",
    "ed_gcs_motor": "Motor response according to GCS at ED",
    "ed_gcs_sum": "GCS score at ED",
    "pre_gcs_motor": "PH motor response according to GCS",
    "pre_gcs_sum": "PH GCS score",
    "res_gos_dischg": "Glascow outcome score at discharge",
    "pt_asa_preinjury": "American Society of Anesthesiologists Class pre-injury",
    "pre_provided": "Level of care given PH",
    "AlarmRePrioritised": "Reprioritisation of trauma code",
    "ed_be_art_NotDone": "Base excess not taken at ED",
    "ed_emerg_proc_other": "Other emergency procedure at ED",
    "ed_emerg_proc": "Emergency procedure at ED",
    "ed_inr_NotDone": "Prothrombin time (international normalized ratio) not taken at ED",
    "ed_intub_type": "Intubation type at ED",
    "ed_intubated": "Was intubated at ED",
    "ed_tta": "Trauma team activated",
    "FirstTraumaDT_NotDone": "Trauma CT not done",
    "hosp_dischg_dest": "Discharge destination",
    "host_care_level": "Highest level of care",
    "host_transfered": "Transferred from/to another hospital",
    "host_vent_days_NotDone": "No days spent on a ventilator",
    "inj_dominant": "Dominant type of injury",
    "inj_intention": "Injury intention",
    "inj_mechanism": "Dominant injury mechanism",
    "pre_card_arrest": "PH cardiac arrest",
    "pre_intub_type": "Type of intubation PH",
    "pre_intubated": "Was intubated PH",
    "pre_transport": "Transport type PH",
    "pt_Gender": "Gender",
    "res_survival": "Mortality",
    "TraumaAlarmAtHospital": "Type of trauma code criteria",
    "TraumaAlarmCriteria": "Trauma code at hospital",
}

swetrau_keys, description = zip(*sorted(zip(format_features.keys(), format_features.values())))

predictor_table = pd.DataFrame(data={"SweTrau code": swetrau_keys, "Description": description})

print(predictor_table.to_latex(index=False))
