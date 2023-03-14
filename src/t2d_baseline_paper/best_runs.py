from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

WANDB_GROUP = ""


@dataclass
class BestRun:
    wandb_group: str
    model: str


run = BestRun(
    wandb_group="prehominid-concordancy",
    model="eternal-pyramid-7790",
)
