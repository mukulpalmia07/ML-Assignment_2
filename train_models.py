"""Train all five classifiers once; save them to model/ and write test_data.csv and output/metrics.csv."""
from app.config.config import settings
from app.config.logger import get_logger
from app.services.data_processor import DataProcessor
from app.services.evaluation_service import Evaluator
from app.services.model_service import ModelService
from app.utilities.file_utils import save_csv

log = get_logger("train")


def main() -> None:
    df = DataProcessor.load()
    log.info("Dataset: %d rows x %d features", len(df), df.shape[1] - 1)
    train_df, test_df = DataProcessor.train_test(df)
    save_csv(test_df, settings.test_data_path)

    service = ModelService.train(*DataProcessor.split_xy(train_df))
    X_test, y_test = DataProcessor.split_xy(test_df, service.features)
    metrics = Evaluator.compare(y_test, service.predict_proba(X_test)).round(4).rename_axis("ML Model Name")
    save_csv(metrics, settings.metrics_path, index=True)
    log.info("Test-set metrics:\n%s", metrics)


if __name__ == "__main__":
    main()
