import pickle
from pathlib import Path

import pandas as pd
from psycop_model_training.model_eval.dataclasses import EvalDataset

from .......users.au484925.t2d-baseline-paper.application.config import best_runs

from .......users.au484925.t2d-baseline-paper.application.config import best_runs


def df_to_eval_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dataframe to EvalDataset."""
    return EvalDataset(
        ids=df["ids"],
        y=df["y"],
        y_hat_probs=df["y_hat_probs"],
        y_hat_int=df["y_hat_int"],
        pred_timestamps=df["pred_timestamps"],
        outcome_timestamps=df["outcome_timestamps"],
        age=df["age"],
        exclusion_timestamps=df["exclusion_timestamps"],
    )


def get_run_item_file_path(wandb_group: str, wandb_run: str, file_name: str) -> Path:
    return Path(
        f"E:/shared_resources/t2d/model_eval/{wandb_group}/{wandb_run}/{file_name}"
    )


def load_eval_dataset(wandb_group: str, wandb_run: str) -> EvalDataset:
    path = get_run_item_file_path(
        wandb_group=wandb_group,
        wandb_run=wandb_run,
        file_name="evaluation_dataset.parquet",
    )
    df = pd.read_parquet(path)
    eval_ds = df_to_eval_dataset(df)
    return eval_ds


def load_file_from_pkl(wandb_group: str, wandb_run: str, file_name: str):
    path = get_run_item_file_path(
        wandb_group=wandb_group,
        wandb_run=wandb_run,
        file_name=file_name,
    )

    with open(path, "rb") as f:
        return pickle.load(f)


def load_fullconfig(wandb_group: str, wandb_run: str) -> pd.DataFrame:
    return load_file_from_pkl(
        wandb_group=wandb_group, wandb_run=wandb_run, file_name="cfg.pkl"
    )


def load_pipe(wandb_group: str, wandb_run: str) -> pd.DataFrame:
    return load_file_from_pkl(
        wandb_group=wandb_group, wandb_run=wandb_run, file_name="pipe.pkl"
    )


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


