import inspect

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import ensemble
import pandas as pd
import numpy as np
from models import random_forest
from feature_reduction import feature_reducer
import trainer
import os

CREATED_BY = "Jeroen Van Hautte"
MODEL_NAME = "Standalone Two Level Classifier"

TEST_INIT = False
TEST_FIT = False
TEST_PREDICT = False

ALL_USERS = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
ALL_ACTIVITIES = np.asarray([1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24])

BASEPATH = "Results/JVH/two_level/basic/"

class TwoLevel(BaseEstimator):
    def __init__(self):
        if (TEST_INIT):
            print "Creating two level model..."
        # Set up classifiers
        self.activityClassifier = random_forest.RF()
        self.userClassifiers = []
        for i in range(12):
            self.userClassifiers.append(random_forest.RF())

    def fit(self, X, y):
        # Retrieve feature set and labels
        train_features = X[:,0:X.shape[1]-1]
        train_activity_labels = X[:,X.shape[1]-1].astype(int).reshape(-1)
        train_subject_labels = y

        if (TEST_FIT):
            temp_trainer = trainer.Trainer("")
            print "Fitting two level model..."

        # Building feature transformers
        ## Activity
        self.activityTransformer = feature_reducer.get_LDA_reducer(
            train_features,
            train_activity_labels,
            20)

        activityFeatures = self.activityTransformer.transform(train_features)

        if (TEST_FIT):
            print "Activity classification:"
            temp_trainer.get_acc_auc(
                self.activityClassifier,
                pd.DataFrame(activityFeatures),
                pd.DataFrame(train_activity_labels),
                os.path.join(BASEPATH + "activity/"))

        self.activityClassifier.fit(activityFeatures, train_activity_labels)

        self.userTransformers = []
        for i in range(len(ALL_ACTIVITIES)):
            current_train_features = train_features[train_activity_labels == ALL_ACTIVITIES[i], :]
            current_train_subject_labels = train_subject_labels[train_activity_labels == ALL_ACTIVITIES[i]]
            self.userTransformers.append(feature_reducer.get_LDA_reducer(
                current_train_features,
                current_train_subject_labels,
                20))

            userFeatures = self.userTransformers[i].transform(current_train_features)

            if(TEST_FIT):
                print "____________________________"
                print "Activity " + str(ALL_ACTIVITIES[i])
                temp_trainer.get_acc_auc(
                    self.activityClassifier,
                    pd.DataFrame(userFeatures),
                    pd.DataFrame(current_train_subject_labels),
                    os.path.join(BASEPATH + "activity"+str(ALL_ACTIVITIES[i])+"/"))

            self.userClassifiers[i].fit(userFeatures, current_train_subject_labels)
        return self

    def predict_proba(self, X):
        if(TEST_PREDICT):
            print "Predicting probabilities..."
        # Retrieve feature set and labels
        train_features = X[:,0:X.shape[1]-1]

        current_features = self.activityTransformer.transform(train_features)
        print(current_features.shape)
        activity_probabilities = self.activityClassifier.predict_proba(current_features)
        if(TEST_PREDICT):
            print "Activity: ",
            print train_features.shape,
            print " -> ",
            print current_features.shape

        user_probabilities = np.zeros((len(activity_probabilities), len(ALL_USERS)))
        for i in range(len(ALL_ACTIVITIES)):
            current_features = self.userTransformers[i].transform(train_features)
            partial_probabilities = self.userClassifiers[i].predict_proba(current_features)

            if (TEST_PREDICT):
                print "Activity " + str(ALL_ACTIVITIES[i]) + " : ",
                print train_features.shape,
                print " -> ",
                print current_features.shape

            current_users = self.userClassifiers[i].estimator.classes_
            for j in range(len(activity_probabilities)):
                if(TEST_PREDICT):
                    if((j+1)%100 == 0):
                        print ".",
                    if(j+1 == len(activity_probabilities)):
                        print ""

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
