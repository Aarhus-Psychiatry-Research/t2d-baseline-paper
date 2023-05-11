from psycop.model_evaluation.binary.time.periodic_plots import (
    plot_roc_auc_by_periodic_time,
)
from t2d.paper_outputs.config import ROBUSTNESS_PATH, RUN_TO_EVAL


def roc_auc_by_cyclic_time():
    print("Plotting AUC by cyclic time")
    eval_ds = RUN_TO_EVAL.get_eval_dataset()

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
