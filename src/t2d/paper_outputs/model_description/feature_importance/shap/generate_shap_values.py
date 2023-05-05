from dataclasses import dataclass
from typing import Sequence

import pandas as pd


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
