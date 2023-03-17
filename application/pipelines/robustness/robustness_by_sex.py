from psycop_model_training.model_eval.base_artifacts.plots.performance_by_sex import (
    plot_performance_by_sex,
)
from t2d_baseline_paper.best_runs import ROBUSTNESS_PATH, best_run
from t2d_baseline_paper.data.load_true_data import load_eval_dataset


def roc_auc_by_sex():
    eval_ds = load_eval_dataset(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run.model,
        custom_columns=["is_female"],
    )

    plot_performance_by_sex(
        eval_dataset=eval_ds,
        save_path=ROBUSTNESS_PATH / "auc_by_sex.png",
    )


if __name__ == "__main__":
    roc_auc_by_sex()
