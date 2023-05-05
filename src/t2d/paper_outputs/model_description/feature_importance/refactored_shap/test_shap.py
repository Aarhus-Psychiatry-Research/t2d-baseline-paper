import pandas as pd
import polars as pl
import pytest
from psycop.test_utils.str_to_df import str_to_df
from sqlalchemy import desc


@pytest.fixture()
def shap_long_df() -> pd.DataFrame:
    return str_to_df(
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


def get_top_i_features_by_shap_variance(shap_long_df: pd.DataFrame, i: int):
    pl_df = pl.from_pandas(shap_long_df)
    feature_stds = pl_df.groupby("feature_name").agg(
        shap_std=pl.col("shap_value").std()
    )

    feature_stds_with_ranks = feature_stds.with_columns(
        shap_std_rank=pl.col("shap_std")
        .rank(method="average", descending=True)
        .cast(pl.Int32)
    )

    return feature_stds_with_ranks.filter(pl.col("shap_std_rank") <= i).to_pandas()


def test_get_top_i_shap(shap_long_df):
    df = get_top_i_features_by_shap_variance(i=2, shap_long_df=shap_long_df)

    # Feature 2 has the largest standard deviation
    assert set(df["feature_name"].unique()) == {"feature_2", "feature_3"}
    assert {"feature_name", "feature_value", "pred_time_index", "shap_value"} == set(
        df.columns
    )


def plot_top_i_shap(shap_long_df: pd.DataFrame, i: int):
    df = get_top_i_features_by_shap_variance(shap_long_df=shap_long_df, i=i)

    pass


def test_plot_top_i_shap(shap_long_df):
    plot_top_i_shap(i=3, shap_long_df=shap_long_df)
