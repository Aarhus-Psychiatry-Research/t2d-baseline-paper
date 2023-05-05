from t2d.paper_outputs.model_description.feature_importance.refactored_shap.get_shap_values import get_long_shap_df, get_shap_explainer_for_best_run


def get_top_ith_feature_df_from_long_shap_df(
        long_shap_df=long_shap_df, i=i
    ):
    pass

def get_top_ith_feature_by_shap_variance(i=1, shap_explainer):
    long_shap_df = get_long_shap_df(shap_explainer=shap_explainer)
    
    top_ith_feature_df = get_top_ith_feature_df_from_long_shap_df(
        long_shap_df=long_shap_df, i=i
    )


def main():
    shap_explainer = get_shap_explainer_for_best_run()

    top_feature_df = get_top_ith_feature_by_shap_variance(
        i=1, shap_explainer=shap_explainer
    )

    # Generate plot
