import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import xgboost as xgb
from sklearn import ensemble
import trainer as t
from feature_reduction import feature_reducer
from models.subject_prediction import random_forest
from sklearn import preprocessing

CREATED_BY = "Jeroen Van Hautte"
MODEL_NAME = "Dual Hybrid Classifier"

TEST_INIT = False
TEST_FIT = False
TEST_PREDICT = False

ALL_USERS = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
ALL_ACTIVITIES = np.asarray([1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24])

BASEPATH = "Results/JVH/hybrid/"

class DualHybrid(BaseEstimator):
    # class1 is expected to be a simple classifier, while class2 is expected to take into account an activity column
    def __init__(self, one_level, two_level):
        # Set up classifiers
        self.one_level_classifier = one_level
        self.two_level_classifier = two_level

    def fit(self, X, y):
        real_features = X[:,0:X.shape[1]-1]
        print "----- HYBRID CLASSIFIER: FITTING 1 ----"
        self.one_level_classifier.fit(real_features, y)
        print "----- HYBRID CLASSIFIER: FITTING 2 ----"
        self.two_level_classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        real_features = X[:,0:X.shape[1]-1]
        print "----- HYBRID CLASSIFIER: PREDICTING 1 ----"
        prob1 = self.one_level_classifier.predict_proba(real_features)
        print "----- HYBRID CLASSIFIER: PREDICTING 2 ----"
        prob2 = self.two_level_classifier.predict_proba(X)
        print "----- HYBRID CLASSIFIER: COMBINING... ----"
        user_probabilities = np.zeros(prob1.shape)
        for i in range(len(prob1)):
            partial_prob1 = prob1[i]
            partial_prob2 = prob2[i]
            if (np.argmax(partial_prob1) == np.argmax(partial_prob2)):
                if(np.amax(partial_prob1) > np.amax(partial_prob2)):
                    user_probabilities[i] = partial_prob1
                else:
                    user_probabilities[i] = partial_prob2
            else:
                user_probabilities[i] = partial_prob1
        return user_probabilities

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1) + np.ones(len(X))

    # This method is used for gridsearch !! needs revision
    def score(self, X, y=None):
        # counts number of values bigger than mean
        return (sum(self.predict(X)))
