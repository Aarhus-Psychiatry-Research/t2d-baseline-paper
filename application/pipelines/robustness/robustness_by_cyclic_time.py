from psycop_model_evaluation.binary.time.periodic_plots import (
    plot_roc_auc_by_periodic_time,
)
from t2d_baseline_paper.best_runs import ROBUSTNESS_PATH, best_run
from t2d_baseline_paper.data.load_true_data import load_eval_dataset


def roc_auc_by_cyclic_time():
    eval_ds = load_eval_dataset(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run.model,
    )

    plot_roc_auc_by_periodic_time(
        eval_dataset=eval_ds,
        bin_period="H",
        save_path=ROBUSTNESS_PATH / "auc_by_hour_of_day.png",
    )

    plot_roc_auc_by_periodic_time(
        eval_dataset=eval_ds,
        bin_period="D",
        save_path=ROBUSTNESS_PATH / "auc_by_day_of_week.png",
    )

    plot_roc_auc_by_periodic_time(
        eval_dataset=eval_ds,
        bin_period="M",
        save_path=ROBUSTNESS_PATH / "auc_by_month_of_year.png",
    )


if __name__ == "__main__":
    roc_auc_by_cyclic_time()
