import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from psycop_model_training.config_schemas.full_config import FullConfigSchema
from sklearn.pipeline import Pipeline


@dataclass
class wandb_run:
    wandb_group: str
    wandb_run: str
    pos_rate: float

    @property
    def eval_dir(self) -> Path:
        return Path(
            f"E:/shared_resources/t2d/model_eval/{self.wandb_group}/{self.wandb_run}"
        )

    @property
    def dataset_dir(self) -> Path:
        config_path = self.get_run_item_file_path(file_name="cfg.json")

        with config_path.open() as f:
            config_str = json.load(f)
            config_dict = json.loads(config_str)

        return Path(config_dict["data"]["dir"])

    @property
    def cfg(self) -> FullConfigSchema:
        return load_file_from_pkl(self.eval_dir / "cfg.pkl")

    @property
    def pipe(self) -> Pipeline:
        return load_file_from_pkl(self.eval_dir / "pipe.pkl")


best_run = wandb_run(
    wandb_group="nonvariably-overpet",
    wandb_run="visionary-armadillo-34",
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
