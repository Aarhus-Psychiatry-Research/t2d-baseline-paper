def get_long_shap_df(shap_explainer):
    """Returns a long dataframe with columns:
    * feature_name (e.g. "age")
    * feature_value (e.g. 31)
    * pred_time_uuid (e.g. "010573-2020-01-01")
    * shap_value (e.g. 0.1)
    Each row represents an observation of a feature at a prediction time.
    """


def get_shap_explainer():
    pass


if __name__ == "__main__":
    shap_explainer = get_shap_explainer()

    long_shap_df = get_long_shap_df(shap_explainer)
