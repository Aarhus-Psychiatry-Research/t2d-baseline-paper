from psycop_model_evaluation.binary.time.timedelta_plots import (
    plot_roc_auc_by_time_from_first_visit,
)
from t2d_baseline_paper.best_runs import ROBUSTNESS_PATH, best_run
from t2d_baseline_paper.data.load_true_data import load_eval_dataset


def roc_auc_by_time_from_first_visit():
    eval_ds = load_eval_dataset(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run.wandb_run,
    )

    plot_roc_auc_by_time_from_first_visit(
        eval_dataset=eval_ds,
        bins=range(0, 37, 3),
        bin_unit="M",
        save_path=ROBUSTNESS_PATH / "auc_by_time_from_first_visit.png",
    )


if __name__ == "__main__":
    roc_auc_by_time_from_first_visit()
