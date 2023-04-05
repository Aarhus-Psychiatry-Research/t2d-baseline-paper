from pathlib import Path
from typing import Sequence

import pandas as pd
from psycop_model_evaluation.descriptive_stats_table import (
    DescriptiveStatsTable,
)
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.data_loader.data_loader import DataLoader
from t2d_baseline_paper.best_runs import TABLES_PATH, best_run
from t2d_baseline_paper.data.load_true_data import load_eval_dataset, load_fullconfig


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
    flattened = load_full_dataset(
        path=Path(
            "E:/shared_resources/t2d/feature_sets/psycop_t2d_adminmanber_features_2023_03_09_18_37"
        )
    )
    additional_columns_names = [
        c
        for c in flattened.columns
        if "pred_f" in c and "max" in c and "_disorders" in c and "730" in c
    ]

    eval_ds = load_eval_dataset(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run.model,
    )

    table = DescriptiveStatsTable(
        eval_dataset=eval_ds,
        additional_columns_df=flattened[[*additional_columns_names, "dw_ek_borger"]],
    ).generate_descriptive_stats_table(
        output_format="df",
        save_path=TABLES_PATH / "descriptive_stats_table.csv",
    )

    table.to_excel(TABLES_PATH / "descriptive_stats_table.xlsx", index=False)

    pass


if __name__ == "__main__":
    descriptive_stats_table()
