import pandas as pd
import polars as pl
from psycop_feature_generation.loaders.raw.load_visits import (
    physical_visits_to_psychiatry,
)
from t2d.feature_generation.eligible_prediction_times.combined_filters import (
    filter_prediction_times_by_eligibility,
)

stepdeltas = []


def get_eligible_prediction_times_as_pandas() -> pd.DataFrame:
    df = pl.from_pandas(
        physical_visits_to_psychiatry(
            timestamps_only=True,
            timestamp_for_output="start",
        ),
    )

    df = filter_prediction_times_by_eligibility(
        df=df,
    )

    return df.to_pandas()


if __name__ == "__main__":
    df = get_eligible_prediction_times_as_pandas()

    for stepdelta in stepdeltas:
        print(
            f"{stepdelta.step_name} dropped {stepdelta.n_dropped}, remaining: {stepdelta.n_after}",
        )

    print(f"Remaining: {df.shape[0]}")

    print(stepdeltas)
