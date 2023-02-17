import pandas as pd
from config import best_runs

from t2d_baseline_paper.data.load_true_data import load_eval_dataset


def load_best_xgb_eval_dataset() -> pd.DataFrame:
    eval_ds = load_eval_dataset(
        wandb_group=best_runs.wandb_group, wandb_run=best_runs.xgboost
    )

    return eval_ds


def load_best_lr_eval_dataset() -> pd.DataFrame:
    eval_ds = load_eval_dataset(
        wandb_group=best_runs.wandb_group, wandb_run=best_runs.logistic_regression
    )

    return eval_ds
