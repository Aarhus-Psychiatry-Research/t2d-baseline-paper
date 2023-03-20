import pickle

import pandas as pd
import shap
from psycop_model_training.utils.col_name_inference import infer_predictor_col_name
from sklearn.pipeline import Pipeline
from zenml.steps import step


def generate_shap_values(features: pd.DataFrame, outcome: pd.DataFrame, pipeline: Pipeline) -> bytes:
    model = pipeline["model"]
    explainer = shap.TreeExplainer(model)  # type: ignore
    shap_values = explainer(features, y=outcome)
    return pickle.dumps(shap_values)
