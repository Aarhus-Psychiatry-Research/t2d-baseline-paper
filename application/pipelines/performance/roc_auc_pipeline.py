from psycop_model_training.model_eval.base_artifacts.plots.roc_auc import plot_auc_roc
from t2d_baseline_paper.best_runs import FIGURES_PATH, best_run
from t2d_baseline_paper.data.load_true_data import load_eval_dataset


def roc_auc_pipeline():
    eval_ds = best_run.get_eval_dataset()

    plot_auc_roc(
        eval_dataset=eval_ds,
        dpi=300,
        save_path=FIGURES_PATH / "auc_roc.png",
    )


if __name__ == "__main__":
    roc_auc_pipeline()
