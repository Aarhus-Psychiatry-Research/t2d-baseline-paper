from config import best_runs
from figures.roc_curve import generate_roc_curve
from figures.shap_values import generate_shap_value_plot

if __name__ == "__main__":
    # generate_roc_curve(use_synth_data=False, best_runs=best_runs)
    generate_shap_value_plot(best_runs=best_runs)
