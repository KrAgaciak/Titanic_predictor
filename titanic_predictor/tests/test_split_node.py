import pytest
import pandas as pd
import sys
from pathlib import Path
from titanic_predictor.src.titanic_predictor.pipelines.data_science.nodes import (
    split_data,
)

project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


class TestSplitNode:
    @pytest.fixture
    def sample_cleaned_data(self):
        return pd.DataFrame(
            {
                "Pclass": [1, 2, 3, 1, 2, 3, 1, 2],
                "Age": [25, 30, 35, 40, 45, 50, 55, 60],
                "SibSp": [0, 1, 0, 1, 0, 1, 0, 1],
                "Parch": [0, 0, 1, 1, 0, 0, 1, 1],
                "Fare": [50, 30, 10, 60, 40, 20, 70, 50],
                "FamilySize": [1, 2, 2, 3, 1, 2, 2, 3],
                "Sex_male": [1, 0, 1, 0, 1, 0, 1, 0],
                "Embarked_Q": [0, 0, 1, 0, 1, 0, 0, 1],
                "Embarked_S": [1, 0, 0, 1, 0, 1, 0, 0],
                "Survived": [1, 0, 1, 0, 1, 0, 1, 0],
            }
        )

    def test_split_data_returns_correct_number_of_datasets(self, sample_cleaned_data):
        result = split_data(
            sample_cleaned_data, test_size=0.25, random_state=42, stratify=True
        )

        X_train, X_test, y_train, y_test = result

        assert len(result) == 4

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_split_data_proportions(self, sample_cleaned_data):
        X_train, X_test, y_train, y_test = split_data(
            sample_cleaned_data, test_size=0.25, random_state=42, stratify=True
        )

        total_samples = len(sample_cleaned_data)
        expected_test_size = int(total_samples * 0.25)

        assert len(X_test) == expected_test_size
        assert len(y_test) == expected_test_size
        assert len(X_train) == total_samples - expected_test_size
        assert len(y_train) == total_samples - expected_test_size

    def test_split_data_no_data_leakage(self, sample_cleaned_data):
        X_train, X_test, y_train, y_test = split_data(
            sample_cleaned_data, test_size=0.25, random_state=42, stratify=True
        )

        train_indices = set(X_train.index)
        test_indices = set(X_test.index)

        assert train_indices.isdisjoint(test_indices)

    def test_split_data_target_distribution(self, sample_cleaned_data):
        X_train, X_test, y_train, y_test = split_data(
            sample_cleaned_data, test_size=0.25, random_state=42, stratify=True
        )

        train_survived_ratio = y_train.mean()
        test_survived_ratio = y_test.mean()
        original_survived_ratio = sample_cleaned_data["Survived"].mean()

        tolerance = 0.1
        assert abs(train_survived_ratio - original_survived_ratio) < tolerance
        assert abs(test_survived_ratio - original_survived_ratio) < tolerance

    def test_split_data_features_consistency(self, sample_cleaned_data):
        X_train, X_test, y_train, y_test = split_data(
            sample_cleaned_data, test_size=0.25, random_state=42, stratify=True
        )

        assert set(X_train.columns) == set(X_test.columns)
        assert "Survived" not in X_train.columns
        assert "Survived" not in X_test.columns
