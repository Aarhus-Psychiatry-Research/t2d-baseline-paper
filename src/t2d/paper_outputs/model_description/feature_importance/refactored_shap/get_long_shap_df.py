import pandas as pd
from t2d.paper_outputs.model_description.feature_importance.shap.generate_shap_values import (
    ShapBundle,
)


def generate_shap_df_for_predictor_col(
    colname: str, predictor_df: pd.DataFrame, shap_values: ShapBundle
):
    colname_index = predictor_df.columns.get_loc(colname)

    df = pd.DataFrame(
        {
            "feature_name": colname,
            "feature_value": predictor_df[colname],
            "pred_time_index": list(range(0, len(predictor_df))),
            "shap_value": shap_values[:, colname_index],
        }
    )

    return df


def get_long_shap_df(shap_bundle: ShapBundle):
    """Returns a long dataframe with columns:
    * feature_name (e.g. "age")
    * feature_value (e.g. 31)
    * pred_time_uuid (e.g. "010573-2020-01-01")
    * shap_value (e.g. 0.1)
    Each row represents an observation of a feature at a prediction time.
    """
    predictor_cols = list(shap_bundle.X.columns)

    dfs = []

    for c in predictor_cols:
        dfs.append(
            generate_shap_df_for_predictor_col(
                colname=c,
                predictor_df=shap_bundle.X,
                shap_values=shap_bundle.shap_values,
            )
        )

    return pd.concat(dfs, axis=0)
