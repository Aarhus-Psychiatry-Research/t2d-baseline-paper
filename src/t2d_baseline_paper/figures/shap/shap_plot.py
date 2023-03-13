import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from t2d_baseline_paper.best_runs import PROJECT_ROOT
from zenml.steps import step


def widen_df_with_limits(ser: pd.Series, widening_factor: float) -> pd.Series:
    ser.iloc[0] = ser.iloc[0] * 1 / widening_factor
    ser.iloc[1] = ser.iloc[1] * widening_factor

    return ser


@step
def plot_shap_scatter(shap_values: bytes) -> None:
    shap_values = pickle.loads(shap_values)

    sns.set(style="whitegrid")

    shap_std = shap_values._numpy_func(  # type: ignore
        fname="std",
        axis=0,
    )

    for i in range(-1, -20, -1):
        shap_for_i: shap._explanation.Explanation = shap_values[:, shap_std.argsort[i]]  # type: ignore

        feature_name = shap_for_i.feature_names

        df = pd.DataFrame(
            {
                feature_name: pd.Series(shap_for_i.data),
                "shap_values": pd.Series(shap_for_i.values),
            },
        )

        n_to_sample = 10_000
        df = df.sample(n=n_to_sample, random_state=42)

        with sns.axes_style("white"):
            x_percentiles, y_percentiles = (
                df[feature_name].quantile([0.05, 0.95]),
                df["shap_values"].quantile([0.05, 0.95]),
            )

            x_percentiles = widen_df_with_limits(x_percentiles, 1.1)
            y_percentiles = widen_df_with_limits(y_percentiles, 1.1)

            dot_alpha = 1 / (n_to_sample / 1_000)

            # Get mean shap_value when feature_name is NaN
            mean_if_nan = df.loc[df[feature_name].isna(), "shap_values"].mean()
            sd_if_nan = df.loc[df[feature_name].isna(), "shap_values"].std()
            lower_if_nan = mean_if_nan - sd_if_nan
            upper_if_nan = mean_if_nan + sd_if_nan

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

            graph = sns.jointplot(
                x=feature_name,
                y="shap_values",
                data=df,
                xlim=x_percentiles,
                ylim=y_percentiles,
                kind="scatter",
                alpha=dot_alpha,
            )

            sns.regplot(
                x=feature_name,
                y="shap_values",
                data=df,
                ax=graph.ax_joint,
                lowess=True,
                color="k",
                scatter=False,
            )

            plt.axhspan(ymin=lower_if_nan, ymax=upper_if_nan, color="orange", alpha=0.2)
            plt.axhline(y=mean_if_nan, color="orange")

            plt.text(
                x=1,
                y=1,
                s="Mean (SD) if val is NaN",
                ha="right",
                va="top",
                transform=plt.gca().transAxes,
                color="orange",
            )

            output_path = (
                PROJECT_ROOT
                / "outputs_for_publishing"
                / "figures"
                / "shap_scatter"
                / f"shap_scatter_{i}.png"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            plt.close()


@step
def plot_beeswarm(shap_values: bytes) -> None:
    shap_values = pickle.loads(shap_values)

    shap.plots.beeswarm(  # type: ignore
        shap_values,
        show=False,
        plot_size=(20, 5),
        max_display=20,
    )

    output_path = (
        PROJECT_ROOT / "outputs_for_publishing" / "figures" / "shap_beeswarm.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)
    plt.close()
