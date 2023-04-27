from psycop_model_evaluation.binary.time.timedelta_plots import (
    plot_time_from_first_positive_to_event,
)

from t2d.evaluation.best_runs import FIGURES_PATH, best_run


def incidence_by_time_until_outcome_pipeline():
    eval_ds = best_run.get_eval_dataset()

    plot_time_from_first_positive_to_event(
        eval_dataset=eval_ds,
        bins=range(0, 37, 1),
        pos_rate=best_run.pos_rate,
        save_path=FIGURES_PATH / "time_from_first_positive_to_event.png",
    )


if __name__ == "__main__":
    incidence_by_time_until_outcome_pipeline()
