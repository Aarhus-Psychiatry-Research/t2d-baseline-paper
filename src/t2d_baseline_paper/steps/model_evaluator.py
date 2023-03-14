import pandas as pd
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.model_eval.model_evaluator import ModelEvaluator
from sklearn.pipeline import Pipeline
from t2d_baseline_paper.best_runs import GENERAL_ARTIFACT_PATH
from t2d_baseline_paper.data.load_true_data import load_eval_dataset, load_fullconfig
from zenml.steps import step


@step
def evaluate_model(
    pipe: Pipeline,
    train_split: pd.DataFrame,
) -> float:
    best_run = best_run.model

    eval_dataset = load_eval_dataset(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run,
    )

    cfg: FullConfigSchema = load_fullconfig(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run,
    )
    cfg.eval.Config.allow_mutation = True
    cfg.eval.lookahead_bins = [0, 90, 180, 270, 360, 450, 540, 630, 720]

    eval_dir_path = GENERAL_ARTIFACT_PATH / "base_model_eval"

    roc_auc = ModelEvaluator(
        cfg=cfg,
        pipe=pipe,
        raw_train_set=train_split,
        eval_ds=eval_dataset,
        eval_dir_path=eval_dir_path,
    ).evaluate()

    return roc_auc
