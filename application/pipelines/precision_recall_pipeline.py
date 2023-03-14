

from t2d_baseline_paper.data.load_true_data import load_eval_dataset
from t2d_baseline_paper.best_runs import PROJECT_ROOT, best_run
from psycop_model_training.model_eval.base_artifacts.plots.precision_recall import plot_precision_recall

def precision_recall_pipeline():
    eval_ds = load_eval_dataset(wandb_group=best_run.wandb_group, wandb_run=best_run.model)
    
    plot_precision_recall(eval_dataset=eval_ds, title="Precision-recall curve", save_path=PROJECT_ROOT / "precision_recall.png")

if __name__ == "__main__":
    precision_recall_pipeline()