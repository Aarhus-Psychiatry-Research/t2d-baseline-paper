
from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.data_loader.utils import (
    load_and_filter_train_and_val_from_cfg,
)
from psycop_model_training.preprocessing.post_split.pipeline import (
    create_post_split_pipeline,
)
from psycop_model_training.training.train_and_predict import train_and_predict
from t2d_baseline_paper.best_runs import best_run
from t2d_baseline_paper.data.load_true_data import load_fullconfig

if __name__ == "__main__":
    pre_split_cfg: FullConfigSchema = load_fullconfig(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run.model,
    )

    pre_split_cfg.preprocessing.pre_split.Config.allow_mutation = True
    pre_split_cfg.preprocessing.pre_split.keep_only_one_outcome_col = True

    dataset = load_and_filter_train_and_val_from_cfg(pre_split_cfg)
    pipe = create_post_split_pipeline(pre_split_cfg)

    eval_dataset = train_and_predict(
        cfg=pre_split_cfg,
        train=dataset.train,
        val=dataset.val,
        pipe=pipe,
        outcome_col_name=outcome_col_name,
        train_col_names=train_col_names,
        n_splits=pre_split_cfg.train.n_splits,
    )
