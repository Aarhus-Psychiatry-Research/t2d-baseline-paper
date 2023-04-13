from psycop_model_training.model_eval.base_artifacts.plots.performance_over_time import (
    plot_recall_by_calendar_time,
)
from t2d_baseline_paper.best_runs import FIGURES_PATH, best_run
from t2d_baseline_paper.data.load_true_data import load_eval_dataset


def incidence_by_time_until_outcome_pipeline():
    eval_ds = load_eval_dataset(
        wandb_group=best_run.wandb_group,
        wandb_run=best_run.wandb_run,
    )

    plot_recall_by_calendar_time(
        eval_dataset=eval_ds,
        bins=range(0, 37, 3),
        pos_rate=[0.05, 0.03, 0.01],
        y_limits=(0, 0.5),
        bin_delta="M",
        save_path=FIGURES_PATH / "sensitivity_by_time_to_event.png",
    )


if __name__ == "__main__":
    incidence_by_time_until_outcome_pipeline()
