from dataclasses import dataclass

WANDB_GROUP = "chapelet-megaloblastic"


from t2d_baseline_paper.globals import BestPerformingRuns

best_runs = BestPerformingRuns(
    wandb_group="chapelet-megaloblastic",
    logistic_regression="revived-pond-2619",
    xgboost="rural-silence-2005",
    lookahead_years=2,
)
