from config import best_runs
from figures.roc_curve import generate_roc_curve
from figures.shap_values import beeswarm_pipeline
from zenml.pipelines import pipeline


@pipeline(enable_cache=True)
def output_pipeline():
    

if __name__ == "__main__":
    # generate_roc_curve(use_synth_data=False, best_runs=best_runs)

    shap_value_plot = beeswarm_pipeline(best_runs=best_runs).run()
