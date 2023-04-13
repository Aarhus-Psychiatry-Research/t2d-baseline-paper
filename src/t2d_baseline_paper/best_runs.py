import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from t2d_baseline_paper.data.load_true_data import load_fullconfig


@dataclass
class wandb_run:
    wandb_group: str
    wandb_run: str
    pos_rate: float

    def get_run_item_file_path(self, file_name: str) -> Path:
        return Path(
            f"E:/shared_resources/t2d/model_eval/{self.wandb_group}/{self.wandb_run}/{file_name}",
        )

    def get_dataset_dir_path(self) -> Path:
        config_path = self.get_run_item_file_path(file_name="cfg.json")
        
        with config_path.open() as f:
            config_str = json.load(f)
            config_dict = json.loads(config_str)
            
        return Path(config_dict["data"]["dir"])


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
