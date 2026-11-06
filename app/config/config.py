import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


class Settings:
    """Project paths and parameters; numeric ones can be overridden in .env."""
    data_path = BASE_DIR / "data" / "dataset.csv"
    test_data_path = BASE_DIR / "test_data.csv"
    model_dir = BASE_DIR / "model"
    metrics_path = BASE_DIR / "output" / "metrics.csv"

    target = "diagnosis"
    labels = ("Benign", "Malignant")  # index = class value
    threshold = 0.5
    test_size = float(os.getenv("TEST_SIZE", 0.2))
    random_state = int(os.getenv("RANDOM_STATE", 42))
    log_level = os.getenv("LOG_LEVEL", "INFO")

    # Display name -> joblib file stem
    models = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "KNN": "knn",
        "Naive Bayes": "naive_bayes",
        "Support Vector Machine": "svm",
    }


settings = Settings()
