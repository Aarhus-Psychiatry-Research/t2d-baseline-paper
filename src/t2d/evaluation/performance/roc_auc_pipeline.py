from psycop.model_evaluation.binary.global_performance.roc_auc import plot_auc_roc
from t2d.evaluation.config import FIGURES_PATH, best_run


def roc_auc_pipeline():
    eval_ds = best_run.get_eval_dataset()

    plot_auc_roc(
        eval_dataset=eval_ds,
        dpi=300,
        save_path=FIGURES_PATH / "auc_roc.png",
    )


if __name__ == "__main__":
    roc_auc_pipeline()
