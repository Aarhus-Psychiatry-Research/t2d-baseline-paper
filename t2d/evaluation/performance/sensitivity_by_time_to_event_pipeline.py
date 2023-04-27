from psycop_model_evaluation.binary.time.timedelta_plots import (
    plot_sensitivity_by_time_to_event,
)

from t2d.evaluation.best_runs import FIGURES_PATH, best_run


def incidence_by_time_until_outcome_pipeline():
    eval_ds = best_run.get_eval_dataset()

    plot_sensitivity_by_time_to_event(
        eval_dataset=eval_ds,
        bins=range(0, 37, 3),
        positive_rates=[0.05, 0.03, 0.01],
        y_limits=(0, 0.5),
        bin_unit="M",
        save_path=FIGURES_PATH / "sensitivity_by_time_to_event.png",
    )


if __name__ == "__main__":
    incidence_by_time_until_outcome_pipeline()
