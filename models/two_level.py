import inspect

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import ensemble
import numpy as np

CREATED_BY = "Jeroen Van Hautte"
MODEL_NAME = "Two Level Estimator"

class TwoLevel(BaseEstimator):
    # Pass one activity estimator and a list of 12 user estimators (all of these have to be pre-trained)
    def __init__(self, activityEstimator, userEstimators):
        self.activityEstimator = activityEstimator
        self.userEstimators = userEstimators

    # Does nothing, as this would force us to use the same feature set for every classifier
    def fit(self, X, y):
        return

    # Expects X to be a list containing 13 feature sets
    def predict_proba(self, X):
        allUsers = [1,2,3,4,5,6,7,8]
        allActivities = ['1','2','3','4','5','6','7','12','13','16','17','24']
        activityProbabilities = self.activityEstimator.predict_proba(X[0]);
        userProbabilities = np.zeros([X[0].shape[0], 8])
        for i in range(12):
            realIndex = np.argwhere(self.activityEstimator.estimator.classes_ == allActivities[i])
            currentPrediction = self.userEstimators[i].predict_proba(X[i+1])
            for j in range(X[i+1].shape[0]):
                for k in range(len(self.userEstimators[i].estimator.classes_)):
                    currentUser = self.userEstimators[i].estimator.classes_[k]
                    userProbabilities[j, np.argwhere(allUsers == currentUser)] += activityProbabilities[j,realIndex]*currentPrediction[j, k]
        return userProbabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        maxProb = np.argmax(probabilities, axis=1)
        return maxProb + np.ones(maxProb.shape)

    # This method is used for gridsearch !! needs revision
    def score(self, X, y=None):
        # counts number of values bigger than mean
        return (sum(self.predict(X)))
