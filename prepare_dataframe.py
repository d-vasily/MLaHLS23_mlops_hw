from typing import List

import pandas as pd


def create_lag_features(
    target: pd.Series, target_name: str, lags: List
) -> pd.DataFrame:
    """
    create lag feature

    :param target: column for feature calculation
    :param target_name: columns name
    :param lags: list of lags for feature

    :return: dataframe that contains only new features
    """
    d = dict()

    for lag in lags:
        d[f"{target_name}_lag{lag}"] = target.shift(lag)

    return pd.DataFrame(data=d)


def create_lag_rolling_features(
    target: pd.Series, target_name: str, lags: List, windows: List, quantiles: List
) -> pd.DataFrame:
    """
    create lag feature

    :param target: column for feature calculation
    :param target_name: columns name
    :param lags: list of lags for feature
    :param windows: list of windows for feature
    :param quantiles: list of quantile for feature

    :return: dataframe that contains only new features
    """
    data_dict = dict()

    for lag in lags:
        for window in windows:
            data_dict[f"{target_name}_lag{lag}_roll{window}_mean"] = (
                target.shift(lag).rolling(window).mean().reset_index(drop=True)
            )
            for quantile in quantiles:
                data_dict[f"{target_name}_lag{lag}_roll{window}_quantile{quantile}"] = (
                    target.shift(lag)
                    .rolling(window)
                    .quantile(quantile)
                    .reset_index(drop=True)
                )

    return pd.DataFrame(data_dict)


def prepare_dataframe(data: pd.DataFrame, features: List, target: str) -> pd.DataFrame:
    """
    create dataframe with features and target

    :param data: dataframe
    :param features: list of columns for feature generation
    :param target: target column

    :return: dataframe with features and target
    """

    l_features = []
    for feature in features:
        l_features.append(
            create_lag_rolling_features(
                data[feature], feature, [1, 5, 22, 260], [3, 5, 22], [0.1, 0.5, 0.9]
            )
        )
    df = pd.concat(l_features, axis=1)
    df["target"] = data[target].shift(-2)
    df = df.dropna().reset_index(drop=True)

    return df
