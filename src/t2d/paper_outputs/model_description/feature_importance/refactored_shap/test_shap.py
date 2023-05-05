from pathlib import Path

import pandas as pd
import plotnine as pn
import polars as pl
import pytest
from psycop.test_utils.str_to_df import str_to_df


@pytest.fixture()
def shap_long_df() -> pl.DataFrame:
    pd_df = str_to_df(
        """feature_name,feature_value,pred_time_index,shap_value
feature_1,1,0,0.1
feature_1,2,1,0.2
feature_1,3,2,0.3
feature_2,1,0,0.2
feature_2,2,1,0.4
feature_2,3,2,0.6
feature_3,1,0,0.3
feature_3,2,1,0.6
feature_3,3,2,0.9
"""
    )

    return pl.from_pandas(pd_df)


def get_top_i_features_by_shap_variance(
    shap_long_df: pl.DataFrame, i: int
) -> pl.DataFrame:
    feature_stds = shap_long_df.groupby("feature_name").agg(
        shap_std=pl.col("shap_value").std()
    )

    feature_stds_with_ranks = feature_stds.with_columns(
        shap_std_rank=pl.col("shap_std")
        .rank(method="average", descending=True)
        .cast(pl.Int32)
    )

    selected_features = feature_stds_with_ranks.filter(i >= pl.col("shap_std_rank"))

    return selected_features.join(shap_long_df, on="feature_name", how="left").drop(
        "shap_std"
    )


def test_get_top_i_shap(shap_long_df: pl.DataFrame):
    df = get_top_i_features_by_shap_variance(i=2, shap_long_df=shap_long_df)

    # Feature 2 has the largest standard deviation
    assert set(df["feature_name"].unique()) == {"feature_2", "feature_3"}
    assert {
        "feature_name",
        "feature_value",
        "pred_time_index",
        "shap_value",
        "shap_std_rank",
    } == set(df.columns)


def plot_shap_for_feature(df: pl.DataFrame, feature_name: str) -> pn.ggplot:
    p = (
        pn.ggplot(df, pn.aes(x="feature_value", y="shap_value"))
        + pn.geom_smooth(method="lm", color="grey")
        + pn.geom_point()
        + pn.theme_minimal()
        + pn.xlab(f"{feature_name}")
        + pn.ylab("SHAP")
    )

    return p


def plot_top_i_shap(shap_long_df: pl.DataFrame, i: int) -> list[pn.ggplot]:
    df = get_top_i_features_by_shap_variance(shap_long_df=shap_long_df, i=i)

    feature_names = df["feature_name"].unique()

    plots = []

    for feature_name in feature_names:
        feature_df = df.filter(pl.col("feature_name") == feature_name)
        p = plot_shap_for_feature(df=feature_df, feature_name=feature_name)

        plots.append(p)

    return plots


def test_plot_top_i_shap(shap_long_df: pl.DataFrame):
    plots = plot_top_i_shap(i=3, shap_long_df=shap_long_df)


def test_top_i_shap(shap_long_df: pl.DataFrame, tmp_path: Path):
    plots = plot_top_i_shap(i=3, shap_long_df=shap_long_df)

    for i, plot in enumerate(plots):
        plot.save(tmp_path / f"plot_{i}.png")

    assert len(list(tmp_path.glob("*.png"))) == 3
