import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from t2d_baseline_paper.best_runs import FIGURES_PATH
from t2d_baseline_paper.feature_name_to_readable import feature_name_to_readable


def widen_df_with_limits(ser: pd.Series, widening_factor: float) -> pd.Series:
    upper_percentile = ser.quantile(0.9)
    lower_percentile = ser.quantile(0.1)
    percentile_range = upper_percentile - lower_percentile

    ser.iloc[0] = ser.iloc[0] - widening_factor * percentile_range
    ser.iloc[1] = ser.iloc[1] + widening_factor * percentile_range

    return ser


def plot_shap_scatter(
    shap_values: bytes,
    n_to_sample: int,
    plot_nan: bool = False,
) -> None:
    shap_values = pickle.loads(shap_values)

    sns.set(style="whitegrid")

    shap_std = shap_values._numpy_func(  # type: ignore
        fname="std",
        axis=0,
    )

    for i in range(-1, -20, -1):
        print(f"Plotting {i}")
        shap_for_i: shap._explanation.Explanation = shap_values[:, shap_std.argsort[i]]  # type: ignore

        feature_name: str = shap_for_i.feature_names  # type: ignore

        df = pd.DataFrame(
            {
                feature_name: pd.Series(shap_for_i.data),
                "shap_values": pd.Series(shap_for_i.values),
            },
        )

        df = df.sample(n=n_to_sample, random_state=42)

        with sns.axes_style("white"):
            # Cut to percentiles
            x_percentiles, y_percentiles = (
                df[feature_name].quantile([0.01, 0.99]),
                df["shap_values"].quantile([0.01, 0.99]),
            )

            x_percentiles = widen_df_with_limits(x_percentiles, 0.2)
            y_percentiles = widen_df_with_limits(y_percentiles, 0.2)
            dot_alpha = 1 / (n_to_sample / 1_000)

            # Create a seaborn scatter plot from the values
            graph = sns.scatterplot(
                data=df,
                x=feature_name,
                y="shap_values",
                alpha=dot_alpha,
            )

            # Add a running average line with SD.
            graph = sns.lineplot(
                data=df,
                x=feature_name,
                y="shap_values",
                estimator="mean",
                errorbar="sd",
                color="orange",
                n_boot=5,
            )

            if plot_nan:
                # Get mean shap_value when feature_name is NaN
                mean_if_nan: float = df.loc[df[feature_name].isna(), "shap_values"].mean()  # type: ignore
                sd_if_nan: float = df.loc[df[feature_name].isna(), "shap_values"].std()  # type: ignore
                lower_if_nan = mean_if_nan - sd_if_nan
                upper_if_nan = mean_if_nan + sd_if_nan

                # Ensure NaN uncertainty band is within the plot boundaries
                y_percentiles.iloc[0] = (
                    lower_if_nan * 0.9
                    if lower_if_nan < y_percentiles.iloc[0]
                    else y_percentiles.iloc[0]
                )
                y_percentiles.iloc[1] = (
                    upper_if_nan * 1.1
                    if upper_if_nan > y_percentiles.iloc[1]
                    else y_percentiles.iloc[1]
                )

                plt.axhspan(
                    ymin=lower_if_nan,
                    ymax=upper_if_nan,
                    color="orange",
                    alpha=0.2,
                )
                plt.axhline(y=mean_if_nan, color="orange")

                plt.text(
                    x=1,
                    y=0.9,
                    s="Mean (SD) if no value found within lookbehind window",
                    ha="right",
                    va="top",
                    transform=plt.gca().transAxes,
                    color="orange",
                )

            # Set the x and y limits
            graph.set_xlim(x_percentiles)
            graph.set_ylim(y_percentiles)

            graph.set_ylabel("SHAP")
            graph.set_xlabel(feature_name_to_readable(feature_name))

            output_path = FIGURES_PATH / "feature_importance" / f"{i}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, dimensions=(5, 5))
            plt.close()
            print(f"Saved plot to {output_path}")
