from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from psycop_model_training.model_eval.dataclasses import EvalDataset


def synth_eval_dataset(noise_to_y_probs: Optional[float] = None) -> EvalDataset:
    """Load synthetic data."""
    csv_path = (
        Path("src")
        / "psycop-model-training"
        / "tests"
        / "test_data"
        / "synth_eval_data.csv"
    )
    df = pd.read_csv(csv_path)
    df = add_age_gender(df)

    # Convert all timestamp cols to datetime
    for col in [col for col in df.columns if "timestamp" in col]:
        df[col] = pd.to_datetime(df[col])

    if noise_to_y_probs:
        df["pred_prob"] = df["pred_prob"] + np.random.normal(
            0, noise_to_y_probs, len(df)
        )

    return EvalDataset(
        ids=df["dw_ek_borger"],
        y=df["label"],
        y_hat_probs=df["pred_prob"],
        y_hat_int=df["pred"],
        pred_timestamps=df["timestamp"],
        outcome_timestamps=df["timestamp_t2d_diag"],
        age=df["age"],
    )


def add_age_gender(df: pd.DataFrame):
    """Add age and gender columns to dataframe.

    Args:
        df (pd.DataFrame): The dataframe to add age
    """

    ids = pd.DataFrame({"dw_ek_borger": df["dw_ek_borger"].unique()})
    ids["age"] = np.random.randint(17, 95, len(ids))
    ids["gender"] = np.where(ids["dw_ek_borger"] > 30_000, "F", "M")

    return df.merge(ids)
