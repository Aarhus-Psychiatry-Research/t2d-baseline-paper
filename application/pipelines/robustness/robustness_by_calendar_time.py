from psycop_model_training.model_eval.base_artifacts.plots.performance_over_time import (
    plot_metric_by_calendar_time,
)
from t2d_baseline_paper.best_runs import ROBUSTNESS_PATH, best_run
from t2d_baseline_paper.data.load_true_data import load_eval_dataset


def roc_auc_by_calendar_time():
    eval_ds = load_eval_dataset(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run.model,

    )

    plot_metric_by_calendar_time(
        eval_dataset=eval_ds,
        bin_period="Q",
        save_path=ROBUSTNESS_PATH / "auc_by_calendar_time.png",
    )


if __name__ == "__main__":
    roc_auc_by_calendar_time()