from psycop_model_training.model_eval.base_artifacts.plots.performance_over_time import (
    plot_metric_by_time_until_diagnosis,
)
from sklearn.metrics import roc_auc_score
from t2d_baseline_paper.best_runs import FIGURES_PATH, ROBUSTNESS_PATH, best_run
from t2d_baseline_paper.data.load_true_data import load_eval_dataset


def roc_auc_by_time_to_event():
    eval_ds = load_eval_dataset(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run.model,
        fraction=0.05,
    )

    plot_metric_by_time_until_diagnosis(
        eval_dataset=eval_ds,
        metric_fn=roc_auc_score,
        y_title="ROC AUC",
        save_path=ROBUSTNESS_PATH / "roc_auc_by_time_to_event.png",
    )


if __name__ == "__main__":
    roc_auc_by_time_to_event()
