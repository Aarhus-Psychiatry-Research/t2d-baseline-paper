from sklearn.pipeline import Pipeline
from t2d_baseline_paper.data.load_true_data import load_pipe
from t2d_baseline_paper.steps.get_train_split import TrainSplitConf
from zenml.steps import step


@step
def pipeline_loader(params: TrainSplitConf) -> Pipeline:
    return load_pipe(
        wandb_group=params.best_runs.wandb_group,
        wandb_run=params.best_runs.model,
    )
