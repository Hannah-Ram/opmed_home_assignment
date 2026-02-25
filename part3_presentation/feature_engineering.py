"""Custom transformers for surgery duration prediction model."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replaces each ID with its count in the training fold (no leakage)."""
    def __init__(self, columns): 
        self.columns = columns
    
    def fit(self, X, y=None):
        self.frequency_maps_ = {col: X[col].value_counts().to_dict() for col in self.columns}
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.frequency_maps_[col]).fillna(0).astype(int)
        return X


class GroupMeanEncoder(BaseEstimator, TransformerMixin):
    """Fold safe target encoding: adds mean per group value, drops raw column."""
    def __init__(self, columns): 
        self.columns = columns
    
    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        self.global_mean_ = float(y_arr.mean())
        self.mean_maps_ = {}
        for col in self.columns:
            tmp = pd.DataFrame({'g': np.asarray(X[col]), 'y': y_arr})
            self.mean_maps_[col] = tmp.groupby('g')['y'].mean().to_dict()
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[f'{col}_mean_duration'] = X[col].map(self.mean_maps_[col]).fillna(self.global_mean_)
            X = X.drop(columns=[col])
        return X
