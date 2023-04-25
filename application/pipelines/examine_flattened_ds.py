from psycop_model_training.config_schemas.full_config import FullConfigSchema
from psycop_model_training.data_loader.data_loader import DataLoader
from t2d_baseline_paper.best_runs import best_run
from t2d_baseline_paper.data.load_true_data import load_fullconfig

if __name__ == "__main__":
    cfg: FullConfigSchema = load_fullconfig(
        wandb_group=best_run.group,
        wandb_run=best_run.name,
    )
    flattened_ds = DataLoader(data_cfg=cfg.data).load_dataset_from_dir(
        split_names="test",
    )
