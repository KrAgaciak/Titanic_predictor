from __future__ import annotations
import time
import os
from typing import Dict, Any
import wandb
import pandas as pd
from autogluon.tabular import TabularPredictor

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

TARGET = "Survived"


def load_raw(df: pd.DataFrame) -> pd.DataFrame:
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    columns_to_drop = ["PassengerId", "Name", "Ticket", "Fare", "Cabin"]
    df_clean = df_clean.drop(
        columns=[col for col in columns_to_drop if col in df_clean.columns]
    )

    df_clean = df_clean.dropna(subset=["Survived"])

    age_median = df_clean["Age"].median()
    df_clean["Age"] = df_clean["Age"].fillna(age_median)

    sibsp_mode = df_clean["SibSp"].mode()[0]
    df_clean["SibSp"] = df_clean["SibSp"].fillna(sibsp_mode)

    parch_mode = df_clean["Parch"].mode()[0]
    df_clean["Parch"] = df_clean["Parch"].fillna(parch_mode)

    embarked_mode = df_clean["Embarked"].mode()[0]
    df_clean["Embarked"] = df_clean["Embarked"].fillna(embarked_mode)

    if "Sex" in df_clean.columns:
        df_clean["Sex"] = df_clean["Sex"].map({"male": 1, "female": 0})

    if "Embarked" in df_clean.columns:
        df_clean["Embarked"] = df_clean["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    return df_clean


def split_data(df: pd.DataFrame, test_size: float, random_state: int, stratify: bool):
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    return X_train, X_test, y_train, y_test


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series, model_params: dict):
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    clf = LogisticRegression(max_iter=model_params.get("max_iter", 1000))
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series):
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    return {"f1": float(f1)}


# ============================================================


def train_autogluon(X_train: pd.DataFrame, y_train: pd.Series, params: Dict[str, Any]):

    seed = params.get("random_seed", 42)
    label = params["autogluon"]["label"]
    time_limit = params["autogluon"].get("time_limit", 120)
    presets = params["autogluon"].get("presets", "medium_quality_faster_train")
    problem_type = params["autogluon"].get("problem_type", "binary")
    eval_metric = params["autogluon"].get("eval_metric", "roc_auc")

    extra_fit_args = params["autogluon"].get("extra_fit_args", {}) or {}

    ag_args_fit = {"seed": seed}

    train_df = X_train.copy()
    train_df[label] = y_train.values

    run = wandb.init(
        project="titanic-team",
        job_type="ag-train",
        config={
            "time_limit": time_limit,
            "presets": presets,
            "problem_type": problem_type,
            "eval_metric": eval_metric,
            "seed": seed,
            **extra_fit_args,
        },
    )

    start = time.time()
    predictor = TabularPredictor(
        label=label,
        problem_type=problem_type,
        eval_metric=eval_metric,
        path="ag_models/",
    ).fit(
        train_df,
        time_limit=time_limit,
        presets=presets,
        verbosity=2,
        ag_args_fit=ag_args_fit,
        **extra_fit_args,
    )
    train_time = time.time() - start

    wandb.log({"train_time_s": train_time})
    run.finish()

    return predictor


def evaluate_autogluon(
    predictor: TabularPredictor, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Any]:

    run = wandb.init(project="titanic-team", job_type="ag-eval")

    label = predictor.label
    test_df = X_test.copy()
    test_df[label] = y_test.values

    perf = predictor.evaluate(test_df, auxiliary_metrics=True)

    try:
        fi = predictor.feature_importance(X_test, y_test=y_test)
    except Exception:
        fi = None

    wandb.log(
        {
            "roc_auc": perf.get("roc_auc"),
            "accuracy": perf.get("accuracy"),
            "log_loss": perf.get("log_loss"),
        }
    )

    run.finish()

    metrics = {"perf": perf, "feature_importance": fi}
    return metrics


def save_best_model(predictor: TabularPredictor, filepath: str):

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    predictor.save("data/06_models/ag_production")

    import joblib

    joblib.dump(predictor, filepath)

    return filepath
