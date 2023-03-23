from collections.abc import Iterable
from typing import Callable

import pandas as pd
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.data_loader.utils import (
    load_and_filter_train_and_val_from_cfg,
)
from psycop_model_training.model_eval.base_artifacts.tables.descriptive_stats_table import (
    DescriptiveStatsTable,
)
from psycop_model_training.model_eval.dataclasses import EvalDataset
from t2d_baseline_paper.best_runs import TABLES_PATH, best_run
from t2d_baseline_paper.data.load_true_data import load_eval_dataset, load_fullconfig


def descriptive_stats_table():
    eval_ds = load_eval_dataset(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run.model,
    )

    table = DescriptiveStatsTable(
        eval_dataset=eval_ds,
    ).generate_descriptive_stats_table(
        output_format="df",
        save_path=TABLES_PATH / "descriptive_stats_table.csv",
    )

    table.to_excel(TABLES_PATH / "descriptive_stats_table.xlsx", index=False)

    pass


if __name__ == "__main__":
    descriptive_stats_table()
