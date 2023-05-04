from psycop.model_evaluation.binary.time.timedelta_plots import (
    plot_roc_auc_by_time_from_first_visit,
)
from t2d.evaluation.config import ROBUSTNESS_PATH, best_run


def roc_auc_by_time_from_first_visit():
    eval_ds = best_run.get_eval_dataset()

    plot_roc_auc_by_time_from_first_visit(
        eval_dataset=eval_ds,
        bins=range(0, 37, 3),
        bin_unit="M",
        save_path=ROBUSTNESS_PATH / "auc_by_time_from_first_visit.png",
    )


if __name__ == "__main__":
    roc_auc_by_time_from_first_visit()
