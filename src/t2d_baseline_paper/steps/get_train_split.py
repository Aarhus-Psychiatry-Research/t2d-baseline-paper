import pandas as pd
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.data_loader.utils import load_and_filter_split_from_cfg
from t2d_baseline_paper.best_runs import BestRun
from t2d_baseline_paper.data.load_true_data import load_fullconfig
from zenml.steps import BaseParameters, step


class TrainSplitConf(BaseParameters):
    best_runs: BestRun




@step
def get_train_split_step(params: TrainSplitConf) -> pd.DataFrame:
    cfg: FullConfigSchema = load_fullconfig(
        wandb_group=params.best_runs.wandb_group,
        wandb_run=params.best_runs.model,
    )
    cfg.Config.allow_mutation = True
    cfg.debug = None

    pass

    df = load_and_filter_split_from_cfg(
        pre_split_cfg=cfg.preprocessing.pre_split, data_cfg=cfg.data, split="train",
    )
    return df


def get_train_split(best_run: BestRun) -> pd.DataFrame:
    cfg: FullConfigSchema = load_fullconfig(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run.model,
    )

    df = load_and_filter_split_from_cfg(
        pre_split_cfg=cfg.preprocessing.pre_split, data_cfg=cfg.data, split="train",
    )

    return df
