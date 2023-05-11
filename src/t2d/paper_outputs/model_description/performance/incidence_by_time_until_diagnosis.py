from psycop.model_evaluation.binary.time.timedelta_plots import (
    plot_time_from_first_positive_to_event,
)
from t2d.paper_outputs.config import FIGURES_PATH, RUN_TO_EVAL


def incidence_by_time_until_outcome_pipeline():
    eval_ds = RUN_TO_EVAL.get_eval_dataset()

    plot_time_from_first_positive_to_event(
        eval_dataset=eval_ds,
        bins=range(0, 37, 1),
        pos_rate=RUN_TO_EVAL.pos_rate,
        save_path=FIGURES_PATH / "time_from_first_positive_to_event.png",
    )


if __name__ == "__main__":
    incidence_by_time_until_outcome_pipeline()
