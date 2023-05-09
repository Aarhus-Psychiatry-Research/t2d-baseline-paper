"""A script for taking the current best model and running it on the test set."""

from psycop.model_training.application_modules.train_model.main import train_model
from t2d.utils.best_runs import Run, RunGroup

if __name__ == "__main__":
    run_group_name = "mistouching-unwontedness"
    BEST_RUN_FROM_CROSSVAL = Run(
        name="townwardspluralistic",
        group=RunGroup(run_group_name),
        pos_rate=0.03,
    )

    cfg = BEST_RUN_FROM_CROSSVAL.cfg
    cfg.project.wandb.Config.allow_mutation = True
    cfg.project.wandb.group = f"{run_group_name}-eval-on-test"
    cfg.data.Config.allow_mutation = True
    cfg.data.splits_for_evaluation = ["test"]

    train_model(cfg=cfg)
