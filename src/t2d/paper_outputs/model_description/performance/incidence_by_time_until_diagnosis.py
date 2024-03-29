import pandas as pd
import plotnine as pn
import polars as pl
from psycop.model_evaluation.binary.time.timedelta_plots import (
    get_time_from_first_positive_to_event_df,
    plot_time_from_first_positive_to_event,
)
from t2d.paper_outputs.config import EVAL_RUN, FIGURES_PATH, PN_THEME


def incidence_by_time_until_outcome_pipeline():
    eval_ds = EVAL_RUN.get_eval_dataset()

    df = pl.DataFrame(
        {
            "y_pred": eval_ds.get_predictions_for_positive_rate(
                desired_positive_rate=EVAL_RUN.pos_rate
            )[0],
            "y": eval_ds.y,
            "patient_id": eval_ds.ids,
            "pred_timestamp": eval_ds.pred_timestamps,
            "time_from_pred_to_event": eval_ds.outcome_timestamps
            - eval_ds.pred_timestamps,
        },
    )

    only_positives = df.filter(pl.col("time_from_pred_to_event").is_not_null())

    plot_df = (
        only_positives.sort("time_from_pred_to_event")
        .groupby("patient_id")
        .head(1)
        .with_columns(
            (pl.col("time_from_pred_to_event").dt.days() / 365.25).alias(
                "years_from_pred_to_event"
            )
        )
    ).to_pandas()

    median_years = plot_df["years_from_pred_to_event"].median()
    annotation_text = f"Median: {str(round(median_years, 1))} years"

    (
        pn.ggplot(plot_df, pn.aes(x="years_from_pred_to_event"))
        + pn.geom_histogram(binwidth=1, fill="orange")
        + pn.xlab("Years from first positive prediction\n to event")
        + pn.scale_x_reverse(
            breaks=range(0, int(plot_df["years_from_pred_to_event"].max() + 1))
        )
        + pn.ylab("n")
        + pn.geom_vline(xintercept=median_years, linetype="dashed", size=1)
        + pn.geom_text(
            pn.aes(x=median_years, y=40),
            label=annotation_text,
            ha="right",
            nudge_x=-0.3,
            size=9,
        )
        + PN_THEME
    ).save(FIGURES_PATH / "time_from_pred_to_event.png", width=5, height=5, dpi=600)


if __name__ == "__main__":
    incidence_by_time_until_outcome_pipeline()
