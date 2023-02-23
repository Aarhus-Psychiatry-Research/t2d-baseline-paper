from config import best_runs
from figures.shap_values import (
    TrainSplitConf,
    generate_shap_values,
    get_train_split,
    pipeline_loader,
    plot_beeswarm,
    plot_shap_scatter,
)
from zenml.pipelines import pipeline


@pipeline(enable_cache=True)
def output_pipeline(
    training_data_loader,
    pipeline_loader,
    shap_generator,
    beeswarm_fn,
    plot_shap_scatter,
):
    train_split = training_data_loader()
    pipe = pipeline_loader()
    shap_values = shap_generator(train_df=train_split, pipeline=pipe)
    beeswarm_fn(shap_values=shap_values)
    plot_shap_scatter(shap_values=shap_values, X=train_split)


if __name__ == "__main__":
    output_pipeline_instance = output_pipeline(
        training_data_loader=get_train_split(TrainSplitConf(best_runs=best_runs)),
        pipeline_loader=pipeline_loader(TrainSplitConf(best_runs=best_runs)),
        shap_generator=generate_shap_values(),
        beeswarm_fn=plot_beeswarm(),
        plot_shap_scatter=plot_shap_scatter(),
    )

    output_pipeline_instance.run(unlisted=True)
