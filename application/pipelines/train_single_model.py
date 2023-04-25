from pathlib import Path

from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.data_loader.utils import (
    load_and_filter_train_and_val_from_cfg,
)
from psycop_model_training.model_eval.model_evaluator import ModelEvaluator
from psycop_model_training.preprocessing.post_split.pipeline import (
    create_post_split_pipeline,
)
from psycop_model_training.training.train_and_predict import train_and_predict
from psycop_model_training.utils.col_name_inference import (
    infer_predictor_col_name,
)
from t2d_baseline_paper.best_runs import best_run
from t2d_baseline_paper.data.load_true_data import load_fullconfig

if __name__ == "__main__":
    cfg: FullConfigSchema = load_fullconfig(
        wandb_group=best_run.group,
        wandb_run=best_run.name,
    )

    cfg.preprocessing.pre_split.Config.allow_mutation = True
    cfg.preprocessing.pre_split.keep_only_one_outcome_col = True

    dataset = load_and_filter_train_and_val_from_cfg(cfg)
    pipe = create_post_split_pipeline(cfg)

    train_col_names = infer_predictor_col_name(df=dataset.train)

    get_eval_dataset = train_and_predict(
        cfg=cfg,
        train=dataset.train,
        val=dataset.val,
        pipe=pipe,
        outcome_col_name="outc_first_diabetes_lab_result_within_1095_days_max_fallback_0_dichotomous",
        train_col_names=train_col_names,
        n_splits=cfg.train.n_splits,
    )

    roc_auc = ModelEvaluator(
        eval_dir_path=Path("E:/shared_resources/eval_debugging/"),
        cfg=cfg,
        pipe=pipe,
        eval_ds=get_eval_dataset,
        raw_train_set=dataset.train,
        upload_to_wandb="offline",
    ).evaluate()

    pass
