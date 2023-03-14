from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class BestRun:
    wandb_group: str
    model: str


best_run = BestRun(
    wandb_group="prehominid-concordancy",
    model="eternal-pyramid-7790",
)

# Get current date as string
date_str = datetime.now().strftime("%Y-%m-%d")

PROJECT_ROOT = Path(__file__).parent.parent.parent
GENERAL_ARTIFACT_PATH = (
    PROJECT_ROOT / "outputs_for_publishing" / date_str / f"{best_run.wandb_group}"
)
FIGURES_PATH = GENERAL_ARTIFACT_PATH / "figures"
TABLES_PATH = GENERAL_ARTIFACT_PATH / "tables"
