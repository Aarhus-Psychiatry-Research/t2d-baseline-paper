from psycop_model_evaluation.binary.subgroups.sex import plot_roc_auc_by_sex
from t2d_baseline_paper.best_runs import ROBUSTNESS_PATH, best_run
from t2d_baseline_paper.data.load_true_data import load_eval_dataset


def roc_auc_by_sex():
    eval_ds = best_run.get_eval_dataset(custom_columns=["is_female"])

    plot_roc_auc_by_sex(
        eval_dataset=eval_ds,
        save_path=ROBUSTNESS_PATH / "auc_by_sex.png",
    )


if __name__ == "__main__":
    roc_auc_by_sex()
