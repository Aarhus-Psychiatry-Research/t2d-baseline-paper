from psycop_model_evaluation.binary.global_performance.precision_recall import (
    plot_precision_recall,
)
from t2d.evaluation.config import FIGURES_PATH, best_run


def precision_recall_pipeline():
    eval_ds = best_run.get_eval_dataset()

    plot_precision_recall(
        eval_dataset=eval_ds,
        title="Precision-recall curve",
        save_path=FIGURES_PATH / "precision_recall.png",
    )


if __name__ == "__main__":
    precision_recall_pipeline()
