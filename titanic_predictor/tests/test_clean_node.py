import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from titanic_predictor.src.titanic_predictor.pipelines.data_science.nodes import (
    basic_clean,
)

project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


class TestCleanNode:
    @pytest.fixture
    def sample_raw_data(self):
        return pd.DataFrame(
            {
                "PassengerId": [1, 2, 3, 4, 5],
                "Survived": [1, 0, 1, 0, np.nan],
                "Pclass": [1, 2, 3, 1, 2],
                "Name": [
                    "John, Mr. Test",
                    "Jane, Miss. Test",
                    "Bob, Mr. Test",
                    "Alice, Miss. Test",
                    "Test, Null. Target",
                ],
                "Sex": ["male", "female", "male", "female", "male"],
                "Age": [25, 30, np.nan, 35, 40],
                "SibSp": [0, 1, np.nan, 2, 0],
                "Parch": [0, 0, 1, np.nan, 0],
                "Ticket": ["A/1", "B/2", "C/3", "D/4", "E/5"],
                "Fare": [50.0, 30.0, 10.0, 60.0, 40.0],
                "Cabin": [np.nan, "C85", np.nan, "E12", "F34"],
                "Embarked": ["S", "C", np.nan, "Q", "S"],
            }
        )

    def test_basic_clean_removes_specified_columns(self, sample_raw_data):
        cleaned_df = basic_clean(sample_raw_data)

        columns_removed = ["PassengerId", "Name", "Ticket", "Fare", "Cabin"]
        for column in columns_removed:
            assert (
                column not in cleaned_df.columns
            ), f"Column {column} should be removed"

    def test_basic_clean_removes_rows_with_null_target(self, sample_raw_data):
        initial_count = len(sample_raw_data)
        cleaned_df = basic_clean(sample_raw_data)

        assert cleaned_df["Survived"].isna().sum() == 0
        assert len(cleaned_df) == initial_count - 1

    def test_basic_clean_fills_missing_values(self, sample_raw_data):
        cleaned_df = basic_clean(sample_raw_data)

        for column in cleaned_df.columns:
            null_count = cleaned_df[column].isna().sum()
            assert null_count == 0, f"Column {column} has {null_count} null values"

    def test_basic_clean_encodes_sex_correctly(self, sample_raw_data):
        cleaned_df = basic_clean(sample_raw_data)

        assert "Sex" in cleaned_df.columns
        assert cleaned_df["Sex"].dtype in [np.int64, int]

        unique_sex_values = set(cleaned_df["Sex"].unique())
        assert unique_sex_values.issubset(
            {0, 1}
        ), f"Sex should be 0 or 1, but got {unique_sex_values}"

        male_mask = sample_raw_data["Sex"] == "male"
        female_mask = sample_raw_data["Sex"] == "female"

        expected_males = len(
            sample_raw_data[male_mask & sample_raw_data["Survived"].notna()]
        )
        expected_females = len(
            sample_raw_data[female_mask & sample_raw_data["Survived"].notna()]
        )

        actual_males = (cleaned_df["Sex"] == 1).sum()
        actual_females = (cleaned_df["Sex"] == 0).sum()

        assert actual_males == expected_males
        assert actual_females == expected_females

    def test_basic_clean_encodes_embarked_correctly(self, sample_raw_data):
        cleaned_df = basic_clean(sample_raw_data)

        assert "Embarked" in cleaned_df.columns
        assert cleaned_df["Embarked"].dtype in [np.int64, int]

        unique_embarked_values = set(cleaned_df["Embarked"].unique())
        assert unique_embarked_values.issubset(
            {0, 1, 2}
        ), f"Embarked should be 0,1,2 but got {unique_embarked_values}"

    def test_basic_clean_preserves_correct_columns(self, sample_raw_data):
        cleaned_df = basic_clean(sample_raw_data)

        expected_columns = [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Embarked",
            "Survived",
        ]
        for column in expected_columns:
            assert column in cleaned_df.columns, f"Column {column} should be present"

    def test_basic_clean_fills_embarked_with_mode(self, sample_raw_data):
        cleaned_df = basic_clean(sample_raw_data)

        assert cleaned_df["Embarked"].isna().sum() == 0

        original_non_null_embarked = sample_raw_data["Embarked"].dropna()
        expected_mode = original_non_null_embarked.mode()[0]
        expected_mode_numeric = (
            0 if expected_mode == "S" else 1 if expected_mode == "C" else 2
        )

        assert (
            expected_mode_numeric in cleaned_df["Embarked"].values
        ), "Embarked should be filled with mode"
