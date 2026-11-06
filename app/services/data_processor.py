import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from app.config.config import settings
from app.utilities.file_utils import save_csv


class DataProcessor:
    """Loads, splits and validates data for the breast cancer classifiers."""

    @staticmethod
    def load() -> pd.DataFrame:
        """UCI Breast Cancer Wisconsin (Diagnostic), cached to data/dataset.csv."""
        if settings.data_path.exists():
            return pd.read_csv(settings.data_path)
        df = load_breast_cancer(as_frame=True).frame.rename(columns={"target": settings.target})
        df[settings.target] = 1 - df[settings.target]  # sklearn encodes malignant as 0; make it the positive class
        save_csv(df, settings.data_path)
        return df

    @staticmethod
    def train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return train_test_split(df, test_size=settings.test_size, random_state=settings.random_state, stratify=df[settings.target])

    @staticmethod
    def split_xy(df: pd.DataFrame, features: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series | None]:
        """Validate and return (X in training column order, y). y is None when the target is absent or incomplete."""
        target = settings.target
        features = features or [c for c in df.columns if c != target]
        if missing := [c for c in features if c not in df.columns]:
            raise ValueError(f"Missing feature columns: {missing}")
        if df.empty:
            raise ValueError("No rows to score.")
        X = df[features].apply(pd.to_numeric, errors="coerce")
        if X.isna().any().any():
            raise ValueError("All feature values must be numeric and non-empty.")
        y = df.get(target)
        if y is None or y.isna().any():
            return X, None
        if not y.isin([0, 1]).all():
            raise ValueError(f"'{target}' must contain only 0 (benign) or 1 (malignant).")
        return X, y.astype(int)
