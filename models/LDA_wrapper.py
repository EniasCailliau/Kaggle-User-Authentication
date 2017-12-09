import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import xgboost as xgb
from sklearn import ensemble
import trainer as t
from feature_reduction import feature_reducer
from sklearn import preprocessing

CREATED_BY = "Jeroen Van Hautte"
MODEL_NAME = "LDA wrapper for any model"

class LDAWrapper(BaseEstimator):
    # class1 is expected to be a simple classifier, while class2 is expected to take into account an activity column
    def __init__(self, estimator):
        # Set up classifiers
        self.estimator = estimator


    def fit(self, X, y):
        self.reducer = feature_reducer.get_LDA_reducer(X,y,20)
        X = np.hstack((self.reducer.transform(X),X))
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_

        return self

    def predict_proba(self, X):
        X = np.hstack((self.reducer.transform(X),X))
        return self.estimator.predict_proba(X)

    def predict(self, X):
        X = np.hstack((self.reducer.transform(X),X))
        return self.estimator.predict(X)

    def score(self, X, y=None):
        # counts number of values bigger than mean
        return (sum(self.predict(X)))
