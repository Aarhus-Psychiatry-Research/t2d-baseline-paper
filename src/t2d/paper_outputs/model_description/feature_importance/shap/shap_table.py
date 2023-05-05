import polars as pl


def get_top_i_shap_values_for_printing(
    shap_long_df: pl.DataFrame,
    i: int,
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
