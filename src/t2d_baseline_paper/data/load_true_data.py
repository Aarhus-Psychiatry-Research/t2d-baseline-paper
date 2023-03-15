import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Memory
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.model_eval.dataclasses import EvalDataset
from sklearn.pipeline import Pipeline

# create a memory cache with a directory to store the cache
memory = Memory(location=Path("E:/shared_resources/t2d/model_eval/model_eval_cache"))


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
        f"E:/shared_resources/t2d/model_eval/{wandb_group}/{wandb_run}/{file_name}",
    )


@memory.cache()
def load_eval_dataset(
    wandb_group: str,
    wandb_run: str,
    fraction: float = 1.0,
) -> EvalDataset:
    path = get_run_item_file_path(
        wandb_group=wandb_group,
        wandb_run=wandb_run,
        file_name="evaluation_dataset.parquet",
    )
    df = pd.read_parquet(path)

    if fraction != 1.0:
        df = df.sample(frac=fraction)

    eval_ds = df_to_eval_dataset(df)

    return eval_ds


def load_file_from_pkl(wandb_group: str, wandb_run: str, file_name: str) -> Any:
    path = get_run_item_file_path(
        wandb_group=wandb_group,
        wandb_run=wandb_run,
        file_name=file_name,
    )

    with path.open("rb") as f:
        return pickle.load(f)


def load_fullconfig(wandb_group: str, wandb_run: str) -> FullConfigSchema:
    return load_file_from_pkl(
        wandb_group=wandb_group,
        wandb_run=wandb_run,
        file_name="cfg.pkl",
    )


def load_pipe(wandb_group: str, wandb_run: str) -> Pipeline:
    return load_file_from_pkl(
        wandb_group=wandb_group,
        wandb_run=wandb_run,
        file_name="pipe.pkl",
    )
