from psycop_model_training.model_eval.base_artifacts.plots.time_from_first_positive_to_event import (
    plot_time_from_first_positive_to_event,
)
from t2d_baseline_paper.best_runs import FIGURES_PATH, best_run
from t2d_baseline_paper.data.load_true_data import load_eval_dataset


def incidence_by_time_until_outcome_pipeline():
    eval_ds = load_eval_dataset(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run.model,
    )

    plot_time_from_first_positive_to_event(
        eval_dataset=eval_ds,
        bins=range(0, 37, 1),
        pos_rate=0.05,
        save_path=FIGURES_PATH / "time_from_first_positive_to_event.png",
    )


if __name__ == "__main__":
    incidence_by_time_until_outcome_pipeline()
