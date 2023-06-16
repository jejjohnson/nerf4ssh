from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class JulianTime(BaseEstimator, TransformerMixin):
    def __init__(
        self,
    ):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:

        X["time"] = X["time"].to_julian_date()

        return X


class TimeDelta(BaseEstimator, TransformerMixin):
    def __init__(
        self, time_min: str = "2005-10-10", time_delta: int = 1, time_unit: str = "s"
    ):
        self.time_min = time_min
        self.time_delta = time_delta
        self.time_unit = time_unit

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        time = X["time"]

        time = (time - np.datetime64(self.time_min)) / np.timedelta64(
            self.time_delta, self.time_unit
        )

        X["time"] = time
        return X

    def inverse_transform(self, X: pd.DataFrame, y=None):
        time = X["time"]

        time = time * np.timedelta64(self.time_delta, self.time_unit) + np.datetime64(
            self.time_min
        )

        X["time"] = time
        return X
