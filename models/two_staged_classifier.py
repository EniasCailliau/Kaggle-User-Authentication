import inspect

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from models import random_forest
from models.activity_prediction import random_forest_user

CREATED_BY = "Enias Cailliau"
MODEL_NAME = "Two Staged Classifier"
ALL_ACTIVITIES = np.asarray([1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24])


class twinzy(BaseEstimator):
    def __init__(self):
        self.activity_classifier = random_forest_user.RandomForest(n_estimators=200, criterion='gini', oob_score=True)
        self.user_classifier = random_forest.RF()

    def fit(self, X, y):
        self.activity_classifier.fit(X, y)
        train_features = X[:, :-1]
        train_activity_labels = X[:, -1]
        train_subject_labels = y

        print("fitting activity_classifier...")
        self.activity_classifier.fit(train_features, train_activity_labels.reshape((-1,)))
        print("fitting user_classifier...")
        self.user_classifier.fit(X, train_subject_labels)

        return self

    def predict_proba(self, X):
        np.set_printoptions(threshold=np.nan)

        X_real = X[:, :-1]
        activity_probabilities = self.activity_classifier.predict_proba(X_real)
        user_probabilities = np.zeros((len(activity_probabilities), 8))
        for index, activity in enumerate(ALL_ACTIVITIES):
            X_with_activity = np.append(X_real, np.full((X_real.shape[0], 1), activity), 1)
            user_probabilities_pre = self.user_classifier.predict_proba(X_with_activity)
            user_probabilities_post = np.repeat(activity_probabilities[:, index].reshape(-1, 1), 8,
                                                axis=1) * user_probabilities_pre
            user_probabilities += user_probabilities_post

        return user_probabilities

    # def predict(self, X):
    #     return self.estimator.predict(X)
    #
    # def score(self, X, y=None):
    #     return (sum(self.predict(X)))
    #
    # def get_params(self, deep=True):
    #     return self.estimator.get_params(deep)

    def get_description(self):
        return "{} created by {}".format(MODEL_NAME, CREATED_BY)
