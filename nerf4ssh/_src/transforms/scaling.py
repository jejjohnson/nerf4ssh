from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Deg2Rad(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str] = None):
        self.columns = columns

    def fit(self, X, y=None):

        if self.columns is None:
            self.columns = X.columns

        return self

    def transform(self, X, y=None):

        X_var = X[self.columns].values

        X_var = np.deg2rad(X_var)

        X = pd.DataFrame(X_var, columns=self.columns)

        return X

    def inverse_transform(self, X, y=None):

        X_var = X[self.columns].values

        X_var = np.rad2deg(X_var)

        X = pd.DataFrame(X_var, columns=self.columns)

        return X


class FixedScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scale: np.ndarray = 1.0, columns: List[str] = None):
        self.columns = columns
        self.scale = scale

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns

        return self

    def transform(self, X, y=None):
        X_var = X[self.columns].values

        X_var *= self.scale

        X = pd.DataFrame(X_var, columns=self.columns)

        return X

    def inverse_transform(self, X, y=None):
        X_var = X[self.columns].values

        X_var /= self.scale

        X = pd.DataFrame(X_var, columns=self.columns)

        return X


class MinMaxFixedScaler(BaseEstimator, TransformerMixin):
    def __init__(
        self, min_val: np.ndarray, max_val: np.ndarray, columns: List[str] = None
    ):
        self.columns = columns
        self.min_val = np.asarray(min_val)
        self.max_val = np.asarray(max_val)

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns

        return self

    def transform(self, X, y=None):

        X_var = X[self.columns].values

        X_var = (X_var - self.min_val) / (self.max_val - self.min_val)

        X = pd.DataFrame(X_var, columns=self.columns)

        return X

    def inverse_transform(self, X, y=None):

        X_var = X[self.columns].values

        X_var = X_var * (self.max_val - self.min_val) + self.min_val

        X = pd.DataFrame(X_var, columns=self.columns)

        return X


class MinMaxDF(BaseEstimator, TransformerMixin):
    def __init__(
        self, columns: List[str] = None, min_val: float = -1, max_val: float = 1
    ):
        self.columns = columns
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X: pd.DataFrame, y=None):

        self.transformer = MinMaxScaler((self.min_val, self.max_val))

        if self.columns is None:
            self.columns = X.columns

        X_var = X[self.columns].values

        self.transformer.fit(X_var)

        return self

    def transform(self, X: pd.DataFrame, y=None):

        # print(f"\n\nType: ", type(X), X.columns)

        X_var = X[self.columns].values
        # X_std = (X - self.min_val) / (self.max_val - self.min_val)
        # X_var = X_std * (self.min_val

        X_var = self.transformer.transform(X_var)

        # print(f"\n\nSHAPE: {X_var.shape}\n\n")
        # print(f"\nDF: {X.shape}\n\n")
        # print(f"\nCOLUMNS: {self.columns}\n\n")
        # print(f"\nX: {X[self.columns].shape}\n\n")
        # X[self.columns].data = X_var
        # print(f"DONE!")
        X = pd.DataFrame(data=X_var, columns=self.columns)

        return X

    def inverse_transform(self, X: pd.DataFrame, y=None):

        X_var = X[self.columns].values

        X_var = self.transformer.inverse_transform(X_var)

        X[self.columns] = X_var

        return X
