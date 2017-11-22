import inspect

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import ensemble
import numpy as np

CREATED_BY = "Jeroen Van Hautte"
MODEL_NAME = "Two Level Estimator"

class RF(BaseEstimator):
    # Pass one activity estimator and a list of 12 user estimators
    def __init__(self, activityEstimator, userEstimators):
        self.activityEstimator = activityEstimator
        self.userEstimators = userEstimators

    # Does nothing, as this would force us to use the same feature set for every classifier
    def fit(self, X, y):
        return

    def predict_proba(self, X):
        allUsers = range(8)
        allActivities = ['1','2','3','4','5','6','7','12','13','16','17','24']
        activityProbabilities = self.activityEstimator.predict_proba(X);
        userProbabilities = np.zeros([X.shape[0], 8])
        for i in range(12):
            realIndex = np.argwhere(self.activityEstimator.classes_ == allActivities[i])
            currentPrediction = self.userEstimators[i].predict_proba(X)
            for j in range(X.shape[0]):
                for currentUser in self.userEstimators[i].classes_:
                    userProbabilities[j, np.argwhere(allUsers == currentUser)] += activityProbabilities[j,realIndex]*currentPrediction[j, np.argwhere(allUsers == currentUser)]
        return userProbabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return probabilities == probabilities.max(axis=1, keepdims=1)

    # This method is used for gridsearch !! needs revision
    def score(self, X, y=None):
        # counts number of values bigger than mean
        return (sum(self.predict(X)))
