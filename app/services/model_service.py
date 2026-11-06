import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.config.config import settings


class ModelService:
    """Holds the five fitted classifiers and scores data with all of them."""

    def __init__(self, models: dict):
        self.models = models
        self.features = list(next(iter(models.values())).feature_names_in_)

    @staticmethod
    def build() -> dict:
        """Scale-sensitive models get a StandardScaler inside the pipeline, so each saved file is self-contained."""
        rs = settings.random_state
        return {
            "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=rs)),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=rs),
            "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
            "Naive Bayes": GaussianNB(),
            "Support Vector Machine": make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(), ensemble=False)),
        }

    @classmethod
    def train(cls, X: pd.DataFrame, y: pd.Series) -> "ModelService":
        settings.model_dir.mkdir(exist_ok=True)
        models = cls.build()
        for name, model in models.items():
            joblib.dump(model.fit(X, y), settings.model_dir / f"{settings.models[name]}.joblib", compress=3)
        return cls(models)

    @classmethod
    def load(cls) -> "ModelService":
        return cls({name: joblib.load(settings.model_dir / f"{stem}.joblib") for name, stem in settings.models.items()})

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """P(malignant) for every row (index) and model (column)."""
        return pd.DataFrame({name: m.predict_proba(X)[:, 1] for name, m in self.models.items()}, index=X.index)
