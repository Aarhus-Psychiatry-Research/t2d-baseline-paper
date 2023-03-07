from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

WANDB_GROUP = ""


@dataclass
class BestPerformingRuns:
    wandb_group: str
    logistic_regression: str
    xgboost: str
    lookahead_years: int


best_runs = BestPerformingRuns(
    wandb_group="sharky-unsheathing",
    logistic_regression="revived-pond-2619",
    xgboost="kind-spaceship-6027",
    lookahead_years=3,
)
