import polars as pl


def get_top_i_shap_values_for_printing(
    shap_long_df: pl.DataFrame,
    i: int,
) -> pl.DataFrame:
    aggregated = shap_long_df.groupby("feature_name").agg(
        pl.col("feature_name").first().alias("Feature"),
        pl.col("shap_value").abs().mean().alias("Mean absolute SHAP"),
    )

    ranked = aggregated.sort(by="Mean absolute SHAP", descending=True).select(
        pl.col("Mean absolute SHAP")
        .rank(method="average", descending=True)
        .cast(pl.Int64)
        .alias("Rank"),
        pl.col("Feature"),
        pl.col("Mean absolute SHAP").round(2).alias("Mean absolute SHAP"),
    )

    return ranked.head(i)
