from psycop_model_training.model_eval.base_artifacts.plots.precision_recall import (
    plot_precision_recall,
)
from t2d_baseline_paper.best_runs import FIGURES_PATH, best_run
from t2d_baseline_paper.data.load_true_data import load_eval_dataset


def precision_recall_pipeline():
    eval_ds = best_run.get_eval_dataset()

    plot_precision_recall(
        eval_dataset=eval_ds,
        title="Precision-recall curve",
        save_path=FIGURES_PATH / "precision_recall.png",
    )


if __name__ == "__main__":
    precision_recall_pipeline()
