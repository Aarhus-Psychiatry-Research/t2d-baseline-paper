from psycop_model_evaluation.binary.subgroups.age import (
    plot_roc_auc_by_age,
)

from t2d.evaluation.best_runs import ROBUSTNESS_PATH, best_run


def roc_auc_by_age():
    eval_ds = best_run.get_eval_dataset()

    plot_roc_auc_by_age(
        eval_dataset=eval_ds,
        bins=[18, *range(20, 80, 5)],
        save_path=ROBUSTNESS_PATH / "auc_by_age.png",
    )


if __name__ == "__main__":
    roc_auc_by_age()
