from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

@dataclass
class BestPerformingRuns:
    wandb_group: str
    logistic_regression: str
    xgboost: str
    lookahead_years: int