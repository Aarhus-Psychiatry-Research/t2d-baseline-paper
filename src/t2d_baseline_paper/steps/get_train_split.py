from application.figures.shap_values import TrainSplitConf
from t2d_baseline_paper.data.load_true_data import load_fullconfig


import pandas as pd
from psycop_model_training.data_loader.utils import load_and_filter_split_from_cfg


@step
def get_train_split(params: TrainSplitConf) -> pd.DataFrame:
    cfg = load_fullconfig(
        wandb_group=params.best_runs.wandb_group, wandb_run=params.best_runs.xgboost
    )

    df = load_and_filter_split_from_cfg(cfg=cfg, split="train")
    return df
