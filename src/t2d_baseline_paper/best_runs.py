import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import pandas as pd
from psycop_model_training.training_output.dataclasses import EvalDataset

# from psycop_model_training.config_schemas.full_config import FullConfigSchema
from sklearn.pipeline import Pipeline
from t2d_baseline_paper.data_loaders.load_true_data import df_to_eval_dataset


@dataclass
class RunGroup:
    name: str

    @property
    def group_dir(self) -> Path:
        return Path(f"E:/shared_resources/t2d/model_eval/{self.name}")


@dataclass
class Run:
    wandb_group: RunGroup
    wandb_run: str
    pos_rate: float

    @property
    def eval_dir(self) -> Path:
        return self.wandb_group.group_dir / self.wandb_run

    @property
    def flattened_ds_dir(self) -> Path:
        config_path = self.eval_dir / "cfg.json"

        with config_path.open() as f:
            config_str = json.load(f)
            config_dict = json.loads(config_str)

        return Path(config_dict["data"]["dir"])

    def get_flattened_split(
        self, split: Literal["train", "test", "val"]
    ) -> pd.DataFrame:
        return pd.read_parquet(self.flattened_ds_dir / f"{split}.parquet")

    @property
    def cfg(self):  # -> FullConfigSchema: # noqa
        return load_file_from_pkl(self.eval_dir / "cfg.pkl")

    @cache  # noqa: B019
    def get_eval_dataset(
        self, custom_columns: Optional[Sequence[str]] = None
    ) -> EvalDataset:
        df = pd.read_parquet(self.eval_dir / "evaluation_dataset.parquet")

        eval_dataset = df_to_eval_dataset(df, custom_columns=custom_columns)

        return eval_dataset

    @property
    def pipe(self) -> Pipeline:
        return load_file_from_pkl(self.eval_dir / "pipe.pkl")


current_group = RunGroup(name="mameluco-cobblestone")

best_run = Run(
    wandb_group=current_group,
    wandb_run="airnwhiteback",
    pos_rate=0.03,
)

# Get current date as string
date_str = datetime.now().strftime("%Y-%m-%d")

PROJECT_ROOT = Path(__file__).parent.parent.parent
GENERAL_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "outputs_for_publishing"
    / date_str
    / f"{best_run.wandb_group}"
    / f"{best_run.wandb_run}"
)
FIGURES_PATH = GENERAL_ARTIFACT_PATH / "figures"
TABLES_PATH = GENERAL_ARTIFACT_PATH / "tables"
ROBUSTNESS_PATH = FIGURES_PATH / "robustness"


def load_file_from_pkl(file_path: Path) -> Any:
    with file_path.open("rb") as f:
        return pickle.load(f)
