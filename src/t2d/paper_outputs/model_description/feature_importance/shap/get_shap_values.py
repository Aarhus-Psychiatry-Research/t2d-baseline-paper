from dataclasses import dataclass

import pandas as pd
import polars as pl
import shap
from sklearn.pipeline import Pipeline
from t2d.paper_outputs.config import best_run
from t2d.utils.cache import mem


@dataclass
class ShapBundle:
    shap_values: list[float]
    X: pd.DataFrame

    def generate_shap_df_for_predictor_col(
        self,
        colname: str,
    ) -> pd.DataFrame:
        colname_index = self.X.columns.get_loc(colname)

        df = pd.DataFrame(
            {
                "feature_name": colname,
                "feature_value": self.X[colname],
                "pred_time_index": list(range(0, len(self.X))),
                "shap_value": self.shap_values[:, colname_index],  # type: ignore
            },
        )

        return df

    def get_long_shap_df(self) -> pd.DataFrame:
        """Returns a long dataframe with columns:
        * feature_name (e.g. "age")
        * feature_value (e.g. 31)
        * pred_time_index (e.g. "010573-2020-01-01")
        * shap_value (e.g. 0.1)
        Each row represents an observation of a feature at a prediction time.
        """
        predictor_cols = list(self.X.columns)

        dfs = []

        for c in predictor_cols:
            dfs.append(
                self.generate_shap_df_for_predictor_col(
                    colname=c,
                ),
            )

        return pd.concat(dfs, axis=0)


def generate_shap_values_from_pipe(
    features: pl.LazyFrame,
    outcome: pl.LazyFrame,
    pipeline: Pipeline,
) -> list[float]:
    numerical_predictors = []

    for c in features.schema:
        if features.schema[c] == pl.Float64 and c.startswith("pred_"):
            numerical_predictors.append(c)

    features = features.with_columns(pl.col(numerical_predictors).round(1).keep_name())

    features_df = features.collect().to_pandas()
    outcome_df = outcome.collect().to_pandas()

    model = pipeline["model"]  # type: ignore
    explainer = shap.TreeExplainer(model)  # type: ignore
    shap_values = explainer.shap_values(features_df, y=outcome_df)
    return shap_values


@mem.cache
def get_shap_bundle_for_best_run(
    run_name: str = best_run.name,
    n_rows: int = 10_000,
    cache_ver: float = 0.1,
) -> ShapBundle:
    print(f"Generating shap values for {run_name}, with cache version {cache_ver}")

    flattened_ds: pl.DataFrame = (
        pl.concat(
            best_run.get_flattened_split_as_lazyframe(split=split) for split in ["train", "val"]  # type: ignore
        )
        .collect()
        .sample(n=n_rows)
    )

    cfg = best_run.cfg
    predictor_cols = [
        c for c in flattened_ds.columns if c.startswith(cfg.data.pred_prefix)
    ]
    outcome_cols = [
        c
        for c in flattened_ds.columns
        if c.startswith(cfg.data.outc_prefix)
        and str(cfg.preprocessing.pre_split.min_lookahead_days) in c
    ]

    pipe = best_run.pipe

    shap_values = generate_shap_values_from_pipe(
        features=flattened_ds.lazy().select(predictor_cols),
        outcome=flattened_ds.lazy().select(outcome_cols),
        pipeline=pipe,  # type: ignore
    )

    return ShapBundle(
        shap_values=shap_values,
        X=flattened_ds.select(predictor_cols).to_pandas(),
    )


def get_top_i_features_by_shap_variance(
    shap_long_df: pl.DataFrame,
    i: int,
) -> pl.DataFrame:
    feature_stds = shap_long_df.groupby("feature_name").agg(
        shap_std=pl.col("shap_value").std(),
    )

    feature_stds_with_ranks = feature_stds.with_columns(
        shap_std_rank=pl.col("shap_std")
        .rank(method="average", descending=True)
        .cast(pl.Int32),
    )

    selected_features = feature_stds_with_ranks.filter(i >= pl.col("shap_std_rank"))

    return selected_features.join(shap_long_df, on="feature_name", how="left").drop(
        "shap_std",
    )


if __name__ == "__main__":
    shap_bundle = get_shap_bundle_for_best_run(
        run_name=best_run.name,
        n_rows=1_000,
        cache_ver=0.1,
    )

    long_shap_df = shap_bundle.get_long_shap_df()

    pass