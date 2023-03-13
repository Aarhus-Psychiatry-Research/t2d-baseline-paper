from t2d_baseline_paper.best_runs import best_runs
from t2d_baseline_paper.figures.shap.generate_shap_values import generate_shap_values
from t2d_baseline_paper.figures.shap.shap_plot import plot_shap_scatter
from t2d_baseline_paper.steps.get_train_split import (
    TrainSplitConf,
    get_train_split,
)
from t2d_baseline_paper.steps.pipeline_loader import pipeline_loader
from zenml.pipelines import pipeline
from zenml.steps.base_step import BaseStep


@pipeline(enable_cache=True)
def shap_pipeline(
    training_data_loader: BaseStep,
    pipeline_loader: BaseStep,
    shap_generator: BaseStep,
    plot_shap_scatter: BaseStep,
):
    train_split = training_data_loader()
    pipe = pipeline_loader()
    shap_values = shap_generator(train_df=train_split, pipeline=pipe)  # type: ignore
    plot_shap_scatter(shap_values=shap_values)  # type: ignore


if __name__ == "__main__":
    SHAP_PIPELINE_INSTANCE = shap_pipeline(
        training_data_loader=get_train_split(TrainSplitConf(best_runs=best_runs)),
        pipeline_loader=pipeline_loader(TrainSplitConf(best_runs=best_runs)),
        shap_generator=generate_shap_values(),
        plot_shap_scatter=plot_shap_scatter(),
    )

    SHAP_PIPELINE_INSTANCE.run(unlisted=True)
