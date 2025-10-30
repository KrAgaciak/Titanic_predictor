from __future__ import annotations
from kedro.pipeline import Pipeline
from titanic_predictor.pipelines.data_science.pipeline import (
    create_pipeline as ds_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    data_science = ds_pipeline()
    return {
        "__default__": data_science,  # domyślnie uruchomi nasz pipeline
        "data_science": data_science,
    }
