from dataclasses import dataclass
from typing import Sequence

import pandas as pd
import polars as pl
import shap
from sklearn.pipeline import Pipeline
from t2d.paper_outputs.config import best_run
from t2d.utils.cache import mem


def generate_shap_values(
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


@dataclass
class ShapBundle:
    shap_values: list[float]
    X: pd.DataFrame

    def get_df(self) -> pd.DataFrame:
        """Get a dataframe where the:
        columns are each of your features
        rows are each of your prediction times
        values are the SHAP value of the feature for that prediction time
        """
        return pd.DataFrame(self.shap_values, columns=self.X.columns)


@mem.cache
def generate_shap_values_for_best_run(
    run_name: str = best_run.name, n_rows: int = 10_000, cache_ver: float = 0.1
) -> ShapBundle:
    print(f"Generating shap values for {run_name}, with cache version {cache_ver}")

    flattened_ds: pl.LazyFrame = (
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

    shap_values = generate_shap_values(
        features=flattened_ds.lazy().select(predictor_cols),
        outcome=flattened_ds.lazy().select(outcome_cols),
        pipeline=pipe,  # type: ignore
    )

    return ShapBundle(
        shap_values=shap_values,
        X=flattened_ds.select(predictor_cols).to_pandas(),
    )
