import pickle

import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from joblib import Memory

mem = Memory(location=".", verbose=0)


@mem.cache
def generate_shap_values(
    features: pd.DataFrame,
    outcome: pd.DataFrame,
    pipeline: Pipeline,
) -> bytes:
    for feature in features.columns:
        if len(features[feature].unique()) > 100:
            features[feature] = features[feature].round(1)

    model = pipeline["model"]
    explainer = shap.TreeExplainer(model)  # type: ignore
    shap_values = explainer(features, y=outcome)
    return pickle.dumps(shap_values)
