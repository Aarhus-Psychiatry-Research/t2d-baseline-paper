from collections.abc import Sequence
from typing import Optional

import pandas as pd
from psycop.model_training.training_output.dataclasses import EvalDataset


def df_to_eval_dataset(
    df: pd.DataFrame,
    custom_columns: Optional[Sequence[str]],
) -> EvalDataset:
    """Convert dataframe to EvalDataset."""
    return EvalDataset(
        ids=df["ids"],
        y=df["y"],
        y_hat_probs=df["y_hat_probs"],
        pred_timestamps=df["pred_timestamps"],
        outcome_timestamps=df["outcome_timestamps"],
        age=df["age"],
        is_female=df["is_female"],
        exclusion_timestamps=df["exclusion_timestamps"],
        pred_time_uuids=df["pred_time_uuids"],
        custom_columns={col: df[col] for col in custom_columns}
        if custom_columns
        else None,
    )
