import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve,
)

from app.config.config import settings


class Evaluator:
    """Turns P(malignant) scores into labels and evaluation metrics."""

    @staticmethod
    def predict(p: pd.Series | pd.DataFrame):
        return (p > settings.threshold).astype(int)  # strict '>' matches sklearn's predict() on ties

    @staticmethod
    def label(p: pd.Series | pd.DataFrame):
        return Evaluator.predict(p).replace(dict(enumerate(settings.labels)))

    @staticmethod
    def metrics(y: pd.Series, p: pd.Series) -> dict:
        pred = Evaluator.predict(p)
        return {
            "Accuracy": accuracy_score(y, pred),
            "AUC": roc_auc_score(y, p) if y.nunique() == 2 else np.nan,  # undefined for a single class
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "MCC": matthews_corrcoef(y, pred),
        }

    @classmethod
    def compare(cls, y: pd.Series, probs: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({name: cls.metrics(y, probs[name]) for name in probs}).T

    @staticmethod
    def roc(y: pd.Series, probs: pd.DataFrame) -> pd.DataFrame:
        """Long-form ROC points (Model, FPR, TPR) for every model."""
        frames = [pd.DataFrame(dict(zip(("FPR", "TPR"), roc_curve(y, probs[n])[:2])), dtype=float).assign(Model=n) for n in probs]
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def confusion(y: pd.Series, pred: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(confusion_matrix(y, pred, labels=[0, 1]), index=settings.labels, columns=settings.labels)

    @staticmethod
    def report(y: pd.Series, pred: pd.Series) -> pd.DataFrame:
        rep = classification_report(y, pred, labels=[0, 1], target_names=settings.labels, output_dict=True, zero_division=0)
        return pd.DataFrame(rep).T.drop(index="accuracy")  # accuracy is shown separately; sklearn repeats it across columns
