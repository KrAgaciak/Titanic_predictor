from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_raw, basic_clean, split_data, train_baseline, evaluate


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(load_raw, inputs="raw_data", outputs="raw_loaded", name="load_raw"),
            node(
                basic_clean,
                inputs="raw_loaded",
                outputs="clean_data",
                name="basic_clean",
            ),
            node(
                split_data,
                inputs=dict(
                    df="clean_data",
                    test_size="params:split.test_size",
                    random_state="params:split.random_state",
                    stratify="params:split.stratify",
                ),
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            ),
            node(
                train_baseline,
                inputs=dict(
                    X_train="X_train", y_train="y_train", model_params="params:model"
                ),
                outputs="model_baseline",
                name="train_baseline",
            ),
            node(
                evaluate,
                inputs=dict(model="model_baseline", X_test="X_test", y_test="y_test"),
                outputs="metrics_baseline",
                name="evaluate",
            ),
        ]
    )
