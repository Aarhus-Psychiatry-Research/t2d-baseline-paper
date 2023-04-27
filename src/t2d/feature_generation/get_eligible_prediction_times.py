# %%
from datetime import datetime

from psycop_feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)

min_date = datetime(year=2013, month=1, day=1)

# %load_ext autoreload
# %autoreload 2

# %%
all_prediction_times = physical_visits_to_psychiatry(
    timestamps_only=True,
    timestamp_for_output="start",
)

# %%
prediction_times_after_min_date = all_prediction_times[
    all_prediction_times["timestamp"] > min_date
]
print(prediction_times_after_min_date.shape[0])


# %%
#########################################
# Filtering based on prevalent diabetes #
#########################################
from t2d.feature_generation.outcome_specification.combined import (
    get_first_diabetes_indicator,
)

first_diabetes_indicator = get_first_diabetes_indicator()

# %%

indicator_before_min_date = first_diabetes_indicator[
    first_diabetes_indicator["timestamp"] < min_date
]

# %%
prediction_times_from_patients_with_diabetes = prediction_times_after_min_date.merge(
    indicator_before_min_date,
    how="inner",
    on="dw_ek_borger",
)

# %% Summarised
print(prediction_times_from_patients_with_diabetes.groupby("source").count()["value"])
print(prediction_times_from_patients_with_diabetes.shape[0])

# %%
#########################
# No prevalent diabetes #
#########################
with_indicator = prediction_times_after_min_date.merge(
    right=indicator_before_min_date,
    how="left",
    on="dw_ek_borger",
)
no_prevalent_diabetes = with_indicator[with_indicator["source"].isna()][
    ["dw_ek_borger", "timestamp_x"]
].rename({"timestamp_x": "timestamp"}, axis=1)
print(no_prevalent_diabetes.shape[0])

# %%
#############################################
# Contacts after  incident diagnosis of T2D #
#############################################
from t2d.feature_generation.outcome_specification.lab_results import (
    get_first_diabetes_lab_result_above_threshold,
)

results_above_threshold = get_first_diabetes_lab_result_above_threshold()
contacts_with_hba1c = no_prevalent_diabetes.merge(
    results_above_threshold,
    on="dw_ek_borger",
    how="left",
    suffixes=("_contact", "_result"),
)
after_incident_diabetes = (
    contacts_with_hba1c["timestamp_contact"] > contacts_with_hba1c["timestamp_result"]
)
not_after_incident_diabetes = contacts_with_hba1c[~after_incident_diabetes]

print(
    f"Visits after incident diabetes: {no_prevalent_diabetes.shape[0] - not_after_incident_diabetes.shape[0]}",
)
print(f"Remaining: {not_after_incident_diabetes.shape[0]}")
# %%
####################################################################
# Contacts within 2 years from entering the Central Denmark Region #
####################################################################
from psycop_feature_generation.application_modules.filter_prediction_times import (
    PredictionTimeFilterer,
)
from psycop_feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)

not_within_two_years_from_move = PredictionTimeFilterer(
    prediction_times_df=not_after_incident_diabetes,
    entity_id_col_name="dw_ek_borger",
    quarantine_timestamps_df=load_move_into_rm_for_exclusion(),
    quarantine_interval_days=730,
    timestamp_col_name="timestamp_contact",
).run_filter()
print(
    f"Within 2 years from move: {not_after_incident_diabetes.shape[0] - not_within_two_years_from_move.shape[0]}",
)
print(f"Remaining: {not_within_two_years_from_move.shape[0]}")

# %%
###################################################
# Whether T2D in the 3 years following prediction #
###################################################
from pathlib import Path

import pandas as pd

dir_path = Path(
    "E:/shared_resources/t2d/feature_sets/psycop_t2d_adminmanber_features_2023_03_22_15_14/",
)
paths = [
    list(dir_path.glob(f"*{split}*.parquet"))[0] for split in ["train", "test", "val"]
]

flattened_dfs = [pd.read_parquet(path) for path in paths]
flattened_dataset = pd.concat(flattened_dfs)

# %%
outcome_col_name = (
    "outc_first_diabetes_lab_result_within_1095_days_max_fallback_0_dichotomous"
)
"outc_first_diabetes_lab_result_within_1095_days_max_fallback_0_dichotomous"
cols_for_outcome_determination = flattened_dataset[
    [outcome_col_name, "dw_ek_borger", "timestamp"]
]
not_within_two_years_from_move = not_within_two_years_from_move.rename(
    {"timestamp_contact": "timestamp"},
    axis=1,
)[["dw_ek_borger", "timestamp"]]
combined = not_within_two_years_from_move.merge(
    cols_for_outcome_determination,
    on=["dw_ek_borger", "timestamp"],
    how="left",
)

combined.groupby(outcome_col_name).count()
# %%
