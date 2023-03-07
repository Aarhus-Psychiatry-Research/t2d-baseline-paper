from config import best_runs

from t2d_baseline_paper.steps.pipeline_loader import pipeline_loader
from t2d_baseline_paper.steps.get_train_split import get_train_split
from t2d_baseline_paper.figures.shap.generate_shap_values import generate_shap_values
from t2d_baseline_paper.steps.get_train_split import (
    TrainSplitConf,
)
from t2d_baseline_paper.figures.shap.shap_plot import plot_shap_scatter
from zenml.pipelines import pipeline


@pipeline(enable_cache=True)
def output_pipeline(
    training_data_loader,
    pipeline_loader,
    shap_generator,
    plot_shap_scatter,
):
    train_split = training_data_loader()
    pipe = pipeline_loader()
    shap_values = shap_generator(train_df=train_split, pipeline=pipe)
    plot_shap_scatter(shap_values=shap_values)
    # beeswarm_fn(shap_values=shap_values)


if __name__ == "__main__":
    OUTPUT_PIPELINE_INSTANCE = output_pipeline(
        training_data_loader=get_train_split(TrainSplitConf(best_runs=best_runs)),
        pipeline_loader=pipeline_loader(TrainSplitConf(best_runs=best_runs)),
        shap_generator=generate_shap_values(),
        plot_shap_scatter=plot_shap_scatter(),
        # beeswarm_fn=plot_beeswarm(),
    )

    OUTPUT_PIPELINE_INSTANCE.run(unlisted=True)
