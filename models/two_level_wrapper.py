import inspect

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import ensemble
import numpy as np
import two_level
from feature_reduction import feature_reducer

PRINT_STATS = False
CREATED_BY = "Jeroen Van Hautte"
MODEL_NAME = "Two Level Wrapper"
# This model wraps around the two level estimator, allowing it to be used more like a regular one
# Setup of the submodels should be done in this class

class TwoLevelWrapper(BaseEstimator):
    # Does nothing
    def __init__(self):
        return

    # Classifiers are set up here, X should be the unreduced feature set, y should be a 2D array (activity | user)
    def fit(self, X, y):
        train_features = X
        train_activity_labels = y[:,0]
        train_subject_labels = y[:, 1]
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

        if(PRINT_STATS):
            trainer.get_acc_auc(activityEstimator, currentFeatures, train_activity_labels, results_location)
        activityEstimator.fit(currentFeatures, train_activity_labels)

        for i in range(12):  # Split this up at a later time for more control
            print "_____________________________________"
            print "Activity " + allActivities[i] + ":"
            currentEstimator = random_forest.RF()
            currentData = train_features.values[train_activity_labels.values == allActivities[i]]
            currentLabels = train_subject_labels.values[train_activity_labels.values == allActivities[i]].reshape(-1)
            currentReducer = feature_reducer.get_LDA_reducer(pd.DataFrame(currentData), pd.DataFrame(currentLabels), 20)
            currentFeatures = currentReducer.transform(train_features)
            self.reducers.append(currentReducer)

            print currentFeatures.shape
            if (PRINT_STATS):
                trainer.get_acc_auc(currentEstimator, pd.DataFrame(currentFeatures), pd.DataFrame(currentLabels),
                                os.path.join("Results", '/'.join(options) + "/" + allActivities[i] + "/"))
            currentEstimator.fit(currentFeatures, currentLabels.ravel())
            userEstimators.append(currentEstimator)
            X_processed.append(currentFeatures)
        self.estimator = two_level.TwoLevel(activityEstimator, userEstimators)
        return self.estimator

    # Expects X to be the unreduced feature set
    def predict_proba(self, X):
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
