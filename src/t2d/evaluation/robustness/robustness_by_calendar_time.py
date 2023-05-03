from psycop.model_evaluation.binary.time.absolute_plots import (
    plot_metric_by_absolute_time,
)
from t2d.evaluation.config import ROBUSTNESS_PATH, best_run


def roc_auc_by_calendar_time():
    eval_ds = best_run.get_eval_dataset()

    plot_metric_by_absolute_time(
        eval_dataset=eval_ds,
        bin_period="Q",
        save_path=ROBUSTNESS_PATH / "auc_by_calendar_time.png",
    )


if __name__ == "__main__":
    roc_auc_by_calendar_time()
