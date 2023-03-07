from application.figures.shap_values import TrainSplitConf
from t2d_baseline_paper.data.load_true_data import load_pipe


from sklearn.pipeline import Pipeline
from zenml.steps import step


@step
def pipeline_loader(params: TrainSplitConf) -> Pipeline:
    return load_pipe(
        wandb_group=params.best_runs.wandb_group, wandb_run=params.best_runs.xgboost
    )
