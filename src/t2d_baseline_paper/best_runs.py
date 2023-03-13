from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

WANDB_GROUP = ""


@dataclass
class BestRun:
    wandb_group: str
    logistic_regression: str
    model: str


run = BestRun(
    wandb_group="sci-adenopharyngeal",
    model="generous-glade-8247",
)
