import pandas as pd
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.data_loader.utils import load_and_filter_split_from_cfg
from t2d_baseline_paper.best_runs import BestRun
from t2d_baseline_paper.data.load_true_data import load_fullconfig
from zenml.steps import BaseParameters, step


class TrainSplitConf(BaseParameters):
    best_runs: BestRun


@step
def get_train_split(params: TrainSplitConf) -> pd.DataFrame:
    cfg: FullConfigSchema = load_fullconfig(
        wandb_group=params.best_runs.wandb_group,
        wandb_run=params.best_runs.model,
    )
    cfg.Config.allow_mutation = True
    cfg.debug = None

    pass

    df = load_and_filter_split_from_cfg(cfg=cfg, split="train")
    return df
