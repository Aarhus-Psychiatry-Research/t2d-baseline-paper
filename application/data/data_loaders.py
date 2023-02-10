import pandas as pd
from config import WANDB_GROUP, BestPerformingModels

from t2d_baseline_paper.data.load_true_data import load_eval_dataset


def load_best_xgb_eval_dataset() -> pd.DataFrame:
    eval_ds = load_eval_dataset(
        wandb_group=WANDB_GROUP, wandb_run=BestPerformingModels().xgboost
    )

    return eval_ds


def load_best_lr_eval_dataset() -> pd.DataFrame:
    eval_ds = load_eval_dataset(
        wandb_group=WANDB_GROUP, wandb_run=BestPerformingModels().logistic_regression
    )

    return eval_ds
