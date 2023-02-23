from dataclasses import dataclass

WANDB_GROUP = ""


from t2d_baseline_paper.globals import BestPerformingRuns

best_runs = BestPerformingRuns(
    wandb_group="sci-adenopharyngeal",
    logistic_regression="revived-pond-2619",
    xgboost="jovial-heart-4924",
    lookahead_years=2,
)
