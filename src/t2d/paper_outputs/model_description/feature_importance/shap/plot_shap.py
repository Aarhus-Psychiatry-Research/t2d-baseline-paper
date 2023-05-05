from pathlib import Path

import plotnine as pn
import polars as pl
from t2d.paper_outputs.model_description.feature_importance.shap.get_shap_values import (
    get_top_i_features_by_shap_variance,
)


def plot_shap_for_feature(df: pl.DataFrame, feature_name: str) -> pn.ggplot:
    p = (
        pn.ggplot(df, pn.aes(x="feature_value", y="shap_value"))
        + pn.geom_point(alpha=0.2, color="blue", shape="+")
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


def save_plots_for_top_i_shap_by_variance(
    shap_long_df: pl.DataFrame,
    i: int,
    save_dir: Path,
) -> Path:
    plots = plot_top_i_shap(i=i, shap_long_df=shap_long_df)

    for i, plot in enumerate(plots):
        print(f"Plotting SHAP panel {i}")
        plot.save(save_dir / f"plot_{i}.jpg")

    return save_dir
