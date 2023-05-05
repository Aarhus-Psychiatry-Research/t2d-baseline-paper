import pandas as pd
import polars as pl
from psycop.test_utils.str_to_df import str_to_df


def get_top_i_shap_values_for_printing(
    shap_long_df: pl.DataFrame, i: int
) -> pl.DataFrame:
    aggregated = shap_long_df.groupby("feature_name").agg(
        pl.col("feature_name").first().alias("Feature"),
        pl.col("shap_value").std().alias("SHAP variance"),
    )

    ranked = aggregated.sort(by="SHAP variance", descending=True).select(
        pl.col("SHAP variance")
        .rank(method="average", descending=True)
        .cast(pl.Int64)
        .alias("Rank"),
        pl.col("Feature"),
        pl.col("SHAP variance").round(2).alias("SHAP variance"),
    )

    return ranked.head(i)


def test_get_top_2_shap_values_for_output():
    expected = str_to_df(
        """Rank,Feature,SHAP variance,
    1,feature_3,0.35,
    2,feature_2,0.14,
    """
    )

    shap_long_df = str_to_df(
        """feature_name,feature_value,pred_time_index,shap_value
feature_1,1,0,0.1
feature_1,2,1,0.2
feature_2,1,0,0.2
feature_2,2,1,0.4
feature_3,1,0,0.5
feature_3,3,2,1.0
""",
    )

    computed = get_top_i_shap_values_for_printing(
        shap_long_df=pl.from_pandas(shap_long_df), i=2
    ).to_pandas()

    # Compare with pandas
    pd.testing.assert_frame_equal(computed, expected, check_dtype=False)
