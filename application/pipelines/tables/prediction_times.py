from psycop_model_training.model_eval.base_artifacts.tables.descriptive_stats_table import (
    DescriptiveStatsTable,
)
from t2d_baseline_paper.best_runs import TABLES_PATH, best_run
from t2d_baseline_paper.data.load_true_data import load_eval_dataset


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
