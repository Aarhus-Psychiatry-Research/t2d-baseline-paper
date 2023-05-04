import polars as pl
from psycop.model_training.config_schemas.full_config import FullConfigSchema
from t2d.model_training.train_model_from_application_module import train_model
from t2d.paper_outputs.config import ESTIMATES_PATH, best_run

if __name__ == "__main__":
    cfg: FullConfigSchema = best_run.cfg

    # Create the dataset with only HbA1c-predictors
    df: pl.LazyFrame = pl.concat(
        best_run.get_flattened_split_as_lazyframe(split) for split in ["train", "val"]  # type: ignore
    )

    # Point the model at that dataset
    cfg.preprocessing.pre_split.Config.allow_mutation = True
    cfg.preprocessing.pre_split.convert_to_boolean = True
    roc_auc = train_model(cfg=cfg)

    ESTIMATES_PATH.mkdir(parents=True, exist_ok=True)

    # Write AUROC
    with (ESTIMATES_PATH / "auroc_for_boolean_predictors.txt").open("a") as f:
        f.write(str(roc_auc))
