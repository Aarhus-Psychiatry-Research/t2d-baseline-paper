from t2d_baseline_paper.globals import BestPerformingRuns

WANDB_GROUP = ""


best_runs = BestPerformingRuns(
    wandb_group="sharky-unsheathing",
    logistic_regression="revived-pond-2619",
    xgboost="kind-spaceship-6027",
    lookahead_years=3,
)
