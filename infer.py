import datetime
import os

import dvc.api
import hydra
import pandas as pd
from catboost import CatBoostRegressor
from omegaconf import DictConfig

import prepare_dataframe


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Функция реализует прогнозирование предобученной моделью.
    """
    dvc.api.DVCFileSystem().get(cfg["paths"]["data"], cfg["paths"]["data"])
    data = pd.read_parquet(cfg["paths"]["data"])

    # generate features
    df = prepare_dataframe.prepare_dataframe(
        data, cfg["prepare_dataframe"]["features"], cfg["prepare_dataframe"]["target"]
    )

    features_list = [col for col in df.columns if "target" not in col]
    next_period_data = df[features_list]

    dvc.api.DVCFileSystem().get(cfg["paths"]["models"], cfg["paths"]["models"])
    model = CatBoostRegressor()
    model.load_model(cfg["paths"]["models"])

    next_period_prediction = pd.Series(model.predict(next_period_data))

    today = datetime.datetime.now().date()

    next_period_prediction.to_csv(
        os.path.join(cfg["paths"]["predictions"], f"prediction_{today}"), index=False
    )

    print("SUCCES")


if __name__ == "__main__":
    main()
