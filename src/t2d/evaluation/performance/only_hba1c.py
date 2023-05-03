import polars as pl
from psycop.model_training.config_schemas.full_config import FullConfigSchema
from t2d.evaluation.config import ESTIMATES_PATH, best_run
from t2d.model_training.train_model_from_application_module import train_model

if __name__ == "__main__":
    cfg: FullConfigSchema = best_run.cfg

    # Create the dataset with only HbA1c-predictors
    df: pl.LazyFrame = pl.concat(
        best_run.get_flattened_split_as_lazyframe(split) for split in ["train", "val"]  # type: ignore
    )

    non_hba1c_pred_cols = [
        c
        for c in df.columns
        if c.startswith(cfg.data.pred_prefix) and ("hba1c" not in c)
    ]

    cols_to_drop = [
        c for c in non_hba1c_pred_cols if "pred_sex" not in c and "pred_age" not in c
    ]

    hba1c_only_df = df.drop(cols_to_drop).collect()

    hba1c_only_dir = best_run.eval_dir / "hba1c_only"
    hba1c_only_dir.mkdir(parents=True, exist_ok=True)
    hba1c_only_path = hba1c_only_dir / "hba1c_only.parquet"
    hba1c_only_df.write_parquet(hba1c_only_path)

    # Point the model at that dataset
    cfg.data.Config.allow_mutation = True
    cfg.data.dir = str(hba1c_only_dir)
    cfg.data.splits_for_training = ["hba1c"]
    roc_auc = train_model(cfg=cfg)

    ESTIMATES_PATH.mkdir(parents=True, exist_ok=True)

    # Write AUROC
    with (ESTIMATES_PATH / "hba1c_only_auroc.txt").open("a") as f:
        f.write(str(roc_auc))
        f.write(str(hba1c_only_df.columns))
