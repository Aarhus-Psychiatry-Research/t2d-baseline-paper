from dataclasses import dataclass
from datetime import datetime

import polars as pl
from psycop_feature_generation.application_modules.filter_prediction_times import (
    PredictionTimeFilterer,
)
from psycop_feature_generation.loaders.raw.load_demographic import birthdays
from psycop_feature_generation.loaders.raw.load_moves import (
    load_move_into_rm_for_exclusion,
)
from psycop_feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)

from t2d.feature_generation.outcome_specification.combined import (
    get_first_diabetes_indicator,
)
from t2d.feature_generation.outcome_specification.lab_results import (
    get_first_diabetes_lab_result_above_threshold,
)

stepdeltas = []
MIN_DATE = datetime(year=2013, month=1, day=1)
MIN_AGE = 18
AGE_COL_NAME = "age"


@dataclass
class StepDelta:
    step_name: str
    n_before_filtering: int
    n_dropped_by_filter: int


def add_stepdelta_manual(step_name: str, n_before: int, n_after: int) -> None:
    stepdeltas.append(
        StepDelta(
            step_name=step_name,
            n_before_filtering=n_before,
            n_dropped_by_filter=n_before - n_after,
        )
    )


def add_stepdelta_from_df(
    step_name: str, before_df: pl.DataFrame, after_df: pl.DataFrame
) -> None:
    stepdeltas.append(
        StepDelta(
            step_name=step_name,
            n_before_filtering=before_df.shape[0],
            n_dropped_by_filter=before_df.shape[0] - after_df.shape[0],
        )
    )


def min_date(df: pl.DataFrame) -> pl.DataFrame:
    after_df = df[df["timestamp"] > MIN_DATE]
    add_stepdelta_from_df(step_name=__name__, before_df=df, after_df=after_df)
    return after_df


def min_age(df: pl.DataFrame) -> pl.DataFrame:
    after_df = df[df[AGE_COL_NAME] >= MIN_AGE]
    add_stepdelta_from_df(step_name=__name__, before_df=df, after_df=after_df)
    return after_df


def without_prevalent_diabetes(df: pl.DataFrame) -> pl.DataFrame:
    first_diabetes_indicator = pl.from_dataframe(get_first_diabetes_indicator())

    indicator_before_min_date = first_diabetes_indicator[
        first_diabetes_indicator["timestamp"] < MIN_DATE
    ]

    prediction_times_from_patients_with_diabetes = df.join(
        indicator_before_min_date,
        on="dw_ek_borger",
        how="inner",
    )

    hit_indicator = prediction_times_from_patients_with_diabetes.groupby(
        "source"
    ).count()

    for indicator in hit_indicator:
        add_stepdelta_manual(
            step_name=indicator["source"],
            n_before=df.shape[0],
            n_after=df.shape[0] - indicator["value"],
        )

    return df.join(
        prediction_times_from_patients_with_diabetes, on="dw_ek_borger", how="anti"
    )


def no_incident_diabetes(df: pl.DataFrame) -> pl.DataFrame:
    results_above_threshold = pl.from_pandas(
        get_first_diabetes_lab_result_above_threshold()
    )

    contacts_with_hba1c = df.join(
        results_above_threshold,
        on="dw_ek_borger",
        how="left",
        suffix="_result",
    )

    after_incident_diabetes = contacts_with_hba1c.filter(
        pl.col("timestamp") > pl.col("timestamp_result")
    )

    not_after_incident_diabetes = contacts_with_hba1c.join(
        after_incident_diabetes, on="dw_ek_borger", how="anti"
    )

    add_stepdelta_from_df(
        step_name=__name__, before_df=df, after_df=not_after_incident_diabetes
    )

    return not_after_incident_diabetes


def washout_move(df: pl.DataFrame) -> pl.DataFrame:
    not_within_two_years_from_move = pl.from_pandas(
        PredictionTimeFilterer(
            prediction_times_df=df.to_pandas(),
            entity_id_col_name="dw_ek_borger",
            quarantine_timestamps_df=load_move_into_rm_for_exclusion(),
            quarantine_interval_days=730,
            timestamp_col_name="timestamp_contact",
        ).run_filter()
    )

    add_stepdelta_from_df(
        step_name=__name__, before_df=df, after_df=not_within_two_years_from_move
    )

    return not_within_two_years_from_move


def add_age(df: pl.DataFrame) -> pl.DataFrame:
    birthday_df = pl.from_pandas(birthdays())

    df = df.join(birthday_df, on="dw_ek_borger", how="inner")
    df = df.with_columns(
        ((pl.col("timestamp") - pl.col("date_of_birth")) / 365.25).alias(AGE_COL_NAME)
    )

    return df


if __name__ == "__main__":
    df = pl.from_pandas(
        physical_visits_to_psychiatry(
            timestamps_only=True,
            timestamp_for_output="start",
        )
    )

    steps = [
        min_date,
        add_age,
        min_age,
        without_prevalent_diabetes,
        washout_move,
    ]

    for step in steps:
        df = step(df)

    print(stepdeltas)
