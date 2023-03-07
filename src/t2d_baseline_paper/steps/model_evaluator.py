from pathlib import Path
from psycop_model_training.model_eval.model_evaluator import ModelEvaluator, EvalDataset
from psycop_model_training.config_schemas.full_config import FullConfigSchema

from sklearn.pipeline import Pipeline
from zenml.steps import BaseParameters, step
import pandas as pd
from t2d_baseline_paper.best_runs import BestPerformingRuns

from t2d_baseline_paper.data.load_true_data import load_eval_dataset, load_fullconfig
from t2d_baseline_paper.best_runs import best_runs

@step
def evaluate_model(
    pipe: Pipeline,
    train_split: pd.DataFrame,
) -> float:
    best_run = best_runs.xgboost

    eval_dataset = load_eval_dataset(
        wandb_group=best_runs.wandb_group,
        wandb_run=best_run,
    )

    cfg = load_fullconfig(wandb_group=best_runs.wandb_group, wandb_run=best_run)
    eval_dir_path = (
        Path(__file__).parent.parent.parent.parent
        / "outputs_for_publishing"
        / "base_model_eval"
    )

    roc_auc = ModelEvaluator(
        cfg=cfg,
        pipe=pipe,
        raw_train_set=train_split,
        eval_ds=eval_dataset,
        eval_dir_path=eval_dir_path,
    ).evaluate()

    return roc_auc
