import os

import hydra
import mlflow
import pandas as pd
from catboost import CatBoostRegressor
from mlflow.models import infer_signature
from omegaconf import DictConfig
from sklearn.metrics import (mean_absolute_percentage_error,
                             median_absolute_error)

import prepare_dataframe


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Training model and logging metrics
    """

    data = pd.read_parquet(cfg["paths"]["data"])

    # generate features and target
    df = prepare_dataframe.prepare_dataframe(
        data, cfg["prepare_dataframe"]["features"], cfg["prepare_dataframe"]["target"]
    )

    features_list = [col for col in df.columns if "target" not in col]

    train_data = df[features_list]
    train_target = df["target"]

    model = CatBoostRegressor(**cfg["catboost_params"])

    model.fit(train_data, train_target)

    model.save_model(os.path.join(cfg["paths"]["models"], "catboost.cbm"), format="cbm")

    if cfg["mlflow"]["logging"]:
        # set tracking server uri for logging
        mlflow.set_tracking_uri(uri=cfg["mlflow"]["logging_uri"])

        # create a new MLflow Experiment
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

        # start an MLflow run
        with mlflow.start_run():
            # log the hyperparameters
            mlflow.log_params(cfg["catboost_params"])

            # calculate metrics
            median_ae = median_absolute_error(train_target, model.predict(train_data))
            mape = mean_absolute_percentage_error(
                train_target, model.predict(train_data)
            )

            # Log metrics
            mlflow.log_metric("median_ae", median_ae)
            mlflow.log_metric("mape", mape)

            # set a tag
            mlflow.set_tag(cfg["mlflow"]["tag_name"], cfg["mlflow"]["tag_value"])

            # Infer the model signature
            signature = infer_signature(train_data, model.predict(train_data))

            # Log the model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=cfg["mlflow"]["artifact_path"],
                signature=signature,
                input_example=train_data,
                registered_model_name=cfg["mlflow"]["registered_model_name"],
            )

    print("SUCCESS")


if __name__ == "__main__":
    main()
