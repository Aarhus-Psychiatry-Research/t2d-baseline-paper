import polars as pl
from t2d.evaluation.config import ESTIMATES_PATH, best_run
from t2d.model_training.train_model_from_application_module import train_model

if __name__ == "__main__":
    # Create the dataset with only HbA1c-predictors
    df: pl.LazyFrame = pl.concat(
        best_run.get_flattened_split_as_lazyframe(split) for split in ["train", "val"]  # type: ignore
    )

    non_hba1c_pred_cols = [
        c for c in df.columns if c.startswith("pred_") and ("hba1c" not in c)
    ]

    hba1c_only_df = df.drop(non_hba1c_pred_cols)

    hba1c_only_path = best_run.eval_dir / "hba1c_only.parquet"
    hba1c_only_df.collect().write_parquet(hba1c_only_path)

    # Point the model at that dataset
    cfg = best_run.cfg
    cfg.data.Config.allow_mutation = True
    cfg.data.dir = str(hba1c_only_path)
    roc_auc = train_model(cfg=cfg)

    ESTIMATES_PATH.mkdir(parents=True, exist_ok=True)

    # Write AUROC
    with (ESTIMATES_PATH / "hba1c_only_auroc.txt").open("w") as f:
        f.write(str(roc_auc))
