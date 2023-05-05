from dataclasses import dataclass

import pandas as pd


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
