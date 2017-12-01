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
MODEL_NAME = "Standalone Two Level Classifier"

TEST_INIT = False
TEST_FIT = False
TEST_PREDICT = False

ALL_USERS = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
ALL_ACTIVITIES = np.asarray([1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24])

BASEPATH = "Results/JVH/two_level/RF2/"

class TwoLevel(BaseEstimator):
    def __init__(self):
        # Set up classifiers
        self.classifiers = []
        self.feature_indices = []
        for i in range(13):
            self.classifiers.append(ensemble.RandomForestClassifier(n_estimators=512, n_jobs=-1, oob_score=True))

    def fit(self, X, y):
        options = ["JVH", "two_level", "RF2"]
        results_location = os.path.join("Results", '/'.join(options) + "/")

        # Retrieve feature set and labels
        train_features = X[:,0:X.shape[1]-1]
        train_activity_labels = X[:,X.shape[1]-1].astype(int).reshape(-1)
        train_subject_labels = y
        self.feature_indices = []

        print "Fitting activity (1)"
        ## Activity
        self.classifiers[0].fit(train_features, train_activity_labels)
        trainer = t.Trainer("")
        trainer.save_estimator(self.classifiers[0], results_location)

        for i in range(len(ALL_ACTIVITIES)):
            print "Fitting " + str(ALL_ACTIVITIES[i])
            current_features = train_features[train_activity_labels == ALL_ACTIVITIES[i]]
            current_labels = train_subject_labels[train_activity_labels == ALL_ACTIVITIES[i]]
            self.classifiers[i+1].fit(current_features, current_labels)

        return self

    def predict_proba(self, X):
        print "Predicting probabilities..."
        # Retrieve feature set and labels
        train_features = X[:,0:X.shape[1]-1]

        # activity_probabilities = self.classifiers[0].predict_proba(train_features)
        acitivities = self.classifiers[0].predict(train_features)
        activity_probabilities = preprocessing.label_binarize(acitivities, np.unique(acitivities))

        user_probabilities = np.zeros((len(activity_probabilities), len(ALL_USERS)))
        for i in range(len(ALL_ACTIVITIES)):
            print "Predicting " + str(ALL_ACTIVITIES[i])

            partial_probabilities = self.classifiers[i+1].predict_proba(train_features)

            current_users = self.classifiers[i+1].classes_
            for j in range(len(activity_probabilities)):
                for cu in current_users:
                    act_prob = activity_probabilities[j, i]
                    partial_prob = partial_probabilities[j, np.argwhere(current_users == cu)]
                    user_probabilities[j, cu-1] += act_prob*partial_prob
        return user_probabilities

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1) + np.ones(len(X))

    # This method is used for gridsearch !! needs revision
    def score(self, X, y=None):
        # counts number of values bigger than mean
        return (sum(self.predict(X)))
