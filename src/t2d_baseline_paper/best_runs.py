from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

WANDB_GROUP = ""


@dataclass
class BestPerformingRuns:
    wandb_group: str
    logistic_regression: str
    xgboost: str


best_runs = BestPerformingRuns(
    wandb_group="sci-adenopharyngeal",
    logistic_regression="revived-pond-2619",
    xgboost="breathless-caress-4374",
)
