import re
from pathlib import Path
from typing import Sequence

import pandas as pd
from numpy import var
from psycop_model_evaluation.descriptive_stats_table import (
    BinaryVariableSpec,
    ContinuousVariableSpec,
    ContinuousVariableToCategorical,
    DatasetSpec,
    VariableGroupSpec,
    create_descriptive_stats_table,
)
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.data_loader.data_loader import DataLoader
from psycop_model_training.preprocessing.pre_split.processors.row_filter import (
    PreSplitRowFilter,
)
from t2d_baseline_paper.best_runs import TABLES_PATH, best_run


def load_full_dataset(
    path: Path,
    splits: Sequence[str] = ("train", "val"),
) -> pd.DataFrame:
    file_names = []
    for split in splits:
        split_paths = list(path.glob(pattern=f"*{split}*"))
        if len(split_paths) == 1:
            file_names += split_paths

    dfs = []

    for f_name in file_names:
        if f_name.suffix == ".parquet":
            df = pd.read_parquet(f_name)
        elif f_name.suffix == ".csv":
            df = pd.read_csv(f_name)

        dfs.append(df)

    return pd.concat(dfs, axis=0)


def descriptive_stats_table():
    run = best_run
    full_dataset = load_full_dataset(path=run.train_dataset_dir)

    # Ensure no rows are dropped because of insufficient lookbehind or lookahead
    pre_split_cfg = run.cfg.preprocessing.pre_split
    pre_split_cfg.Config.allow_mutation = True
    pre_split_cfg.min_lookahead_days = None
    pre_split_cfg.lookbehind_combination = None

    preprocessed_dataset = PreSplitRowFilter(
        pre_split_cfg=pre_split_cfg, data_cfg=run.cfg.data
    ).run_filter(dataset=full_dataset)

    preprocessed_dataset["first_visit"] = preprocessed_dataset.groupby("dw_ek_borger")[
        "timestamp"
    ].transform("min")
    preprocessed_dataset["time_from_first_visit_to_t2d"] = (
        preprocessed_dataset["timestamp_first_diabetes_lab_result"]
        - preprocessed_dataset["first_visit"]
    ).dt.days

    patients_group = VariableGroupSpec(
        title="Patients",
        group_column_name="dw_ek_borger",
        add_total_row=True,
        variable_specs=[
            BinaryVariableSpec(
                variable_title="Female sex",
                variable_df_col_name="pred_sex_female",
                within_group_aggregation="max",
                positive_class=1,
            ),
            BinaryVariableSpec(
                variable_title="Incident T2D",
                variable_df_col_name="outc_first_diabetes_lab_result_within_1095_days_max_fallback_0_dichotomous",
                within_group_aggregation="max",
                positive_class=1,
            ),
            ContinuousVariableSpec(
                variable_title="Days from first contact to first outcome",
                variable_df_col_name="time_from_first_visit_to_t2d",
                within_group_aggregation="max",
                aggregation_function="median",
                variance_measure="iqr",
            ),
        ],
    )

    # Get diagnosis columns
    pattern = re.compile(r"pred_f\d_disorders")
    fx_disorders = sorted(
        [
            c
            for c in preprocessed_dataset.columns
            if pattern.search(c) and "max" in c and "730" in c
        ]
    )

    disorder_specs = []
    for i, column_name in enumerate(fx_disorders):
        spec = BinaryVariableSpec(
            variable_title=f"F{i}", variable_df_col_name=column_name, positive_class=1
        )
        disorder_specs.append(spec)

    contacts_group = VariableGroupSpec(
        title="Contacts",
        group_column_name=None,
        add_total_row=True,
        variable_specs=[
            ContinuousVariableSpec(
                variable_title="Age at contact",
                variable_df_col_name="pred_age_in_years",
                aggregation_function="mean",
                variance_measure="std",
                n_decimals=None,
            ),
            ContinuousVariableToCategorical(
                variable_title="Age at contact",
                variable_df_col_name="pred_age_in_years",
                bins=[18, 24, 34, 44, 54, 64, 74],
                n_decimals=None,
                bin_decimals=None,
            ),
            *disorder_specs,
            BinaryVariableSpec(
                variable_title="Incident type 2 diabetes within 3 years",
                variable_df_col_name="outc_first_diabetes_lab_result_within_1095_days_max_fallback_0_dichotomous",
                positive_class=1,
            ),
        ],
    )

    datasets = [DatasetSpec(title="Train", df=preprocessed_dataset)]

    descriptive_table = create_descriptive_stats_table(
        variable_group_specs=[patients_group, contacts_group],
        datasets=datasets,
    )

    descriptive_table.to_csv(TABLES_PATH / "descriptive_stats_table.csv", index=False)


if __name__ == "__main__":
    descriptive_stats_table()
