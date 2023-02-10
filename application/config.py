from dataclasses import dataclass

WANDB_GROUP = "chapelet-megaloblastic"


@dataclass
class BestPerformingModels:
    logistic_regression: str = "revived-pond-2619"
    xgboost: str = "rural-silence-2005"
    lookahead_years: int = 2
