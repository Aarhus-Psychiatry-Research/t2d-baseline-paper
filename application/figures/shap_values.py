import matplotlib.pyplot as plt
import pandas as pd
import shap
from psycop_model_training.data_loader.utils import load_and_filter_split_from_cfg
from psycop_model_training.utils.col_name_inference import infer_predictor_col_name
from zenml.pipelines import pipeline
from zenml.steps import BaseParameters, Output, step

from t2d_baseline_paper.data.load_true_data import load_fullconfig, load_pipe
from t2d_baseline_paper.globals import PROJECT_ROOT, BestPerformingRuns


class ShapValueConf(BaseParameters):
    best_runs: BestPerformingRuns
    dpi: int = 300


@pipeline(enable_cache=True)
def beeswarm_step(best_runs: BestPerformingRuns, train_loader, dpi: int = 300):
    # Load cfg from pickle
    df = get_train_split(best_runs)

    pred_col_names = infer_predictor_col_name(df)
    X = df[pred_col_names]

    pipe = load_pipe(wandb_group=best_runs.wandb_group, wandb_run=best_runs.xgboost)
    model = pipe["model"]

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    plt.figure(figsize=(5, 5), dpi=dpi)
    shap.plots.beeswarm(shap_values, show=False)

    OUTPUT_PATH = (
        PROJECT_ROOT / "outputs_for_publishing" / "figures" / "shap_beeswarm.png"
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(OUTPUT_PATH)
    plt.close


@step
def get_train_split(best_runs) -> Output(df=pd.DataFrame):
    cfg = load_fullconfig(
        wandb_group=best_runs.wandb_group, wandb_run=best_runs.xgboost
    )

    df = load_and_filter_split_from_cfg(cfg=cfg, split="train")
    return df
