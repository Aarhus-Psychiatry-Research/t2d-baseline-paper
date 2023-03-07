from config import best_runs
from figures.shap_values import (
    TrainSplitConf,
    generate_shap_values,
    get_train_split,
    pipeline_loader,
    plot_shap_scatter,
)
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
    output_pipeline_instance = output_pipeline(
        training_data_loader=get_train_split(TrainSplitConf(best_runs=best_runs)),
        pipeline_loader=pipeline_loader(TrainSplitConf(best_runs=best_runs)),
        shap_generator=generate_shap_values(),
        plot_shap_scatter=plot_shap_scatter(),
        # beeswarm_fn=plot_beeswarm(),
    )

    output_pipeline_instance.run(unlisted=True)
