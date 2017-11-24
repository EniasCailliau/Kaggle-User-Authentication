import inspect
import trainer

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import ensemble
import numpy as np
import two_level
from feature_reduction import feature_reducer
from models import random_forest
import pandas as pd

CREATED_BY = "Jeroen Van Hautte"
MODEL_NAME = "Two Level Wrapper"
# This model wraps around the two level estimator, allowing it to be used more like a regular one
# Setup of the submodels should be done in this class

class TwoLevelWrapper(BaseEstimator):
    # Does nothing
    def __init__(self, results_location):
        self.trainer = trainer.Trainer("")
        self.results_location = results_location
        return

    # Classifiers are set up here, X should contain the unreduced feature set and the activity labels, y should be a 1D array containing only user labels
    def fit(self, X, y):
        train_features = X[:, 0:len(X[0])-1]
        train_activity_labels = X[:, len(X[0])-1]
        train_subject_labels = y
        allUsers = [1, 2, 3, 4, 5, 6, 7, 8]
        allActivities = ['1', '2', '3', '4', '5', '6', '7', '12', '13', '16', '17', '24']

        activityEstimator = random_forest.RF()

        # assemble lists
        X_processed = []
        userEstimators = []
        self.reducers = []

        currentReducer = feature_reducer.get_LDA_reducer(train_features, train_activity_labels, 20)
        currentFeatures = currentReducer.transform(train_features)

        self.reducers.append(currentReducer)
        X_processed.append(currentFeatures)
        activityEstimator.fit(currentFeatures, train_activity_labels)

        for i in range(12):  # Split this up at a later time for more control
            print "_____________________________________"
            print "Activity " + allActivities[i] + ":"
            currentEstimator = random_forest.RF()
            currentData = train_features[train_activity_labels == allActivities[i]]
            currentLabels = train_subject_labels[train_activity_labels == allActivities[i]].reshape(-1)
            currentReducer = feature_reducer.get_LDA_reducer(pd.DataFrame(currentData), pd.DataFrame(currentLabels), 20)
            currentFeatures = currentReducer.transform(train_features[train_activity_labels == allActivities[i]])
            self.reducers.append(currentReducer)

            print currentFeatures.shape
            currentEstimator.fit(currentFeatures, currentLabels.ravel())
            userEstimators.append(currentEstimator)
            X_processed.append(currentFeatures)
        self.estimator = two_level.TwoLevel(activityEstimator, userEstimators)
        return self.estimator

    # Expects X to be the unreduced feature set with an added column that will be discarded
    def predict_proba(self, X):
        X = X[:, 0:len(X[0]) - 1]
        X_processed = []
        for i in range(13):
            X_processed.append(self.reducers[i].transform(X))
        return self.estimator.predict_proba(X_processed)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        maxProb = np.argmax(probabilities, axis=1)
        return maxProb + np.ones(maxProb.shape)

    # This method is used for gridsearch !! needs revision
    def score(self, X, y=None):
        # counts number of values bigger than mean
        return (sum(self.predict(X)))
