from pathlib import Path

import pandas as pd
from psycop_model_training.model_eval.dataclasses import EvalDataset


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


def load_best_xgb_eval_dataset() -> pd.DataFrame:
    eval_ds = load_eval_dataset(
        wandb_group="chapelet-megaloblastic", wandb_run="rural-silence-2005"
    )

    return eval_ds


def load_best_lr_eval_dataset() -> pd.DataFrame:
    eval_ds = load_eval_dataset(
        wandb_group="chapelet-megaloblastic", wandb_run="revived-pond-2619"
    )

    return eval_ds


def load_eval_dataset(wandb_group: str, wandb_run: str) -> pd.DataFrame:
    path = Path(
        f"E:/shared_resources/t2d/model_eval/{wandb_group}/{wandb_run}/evaluation_dataset.parquet"
    )
    df = pd.read_parquet(path)
    eval_ds = df_to_eval_dataset(df)
    return eval_ds
