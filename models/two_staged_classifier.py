import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

ALL_ACTIVITIES = np.asarray([1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24])


class simple_twinzy(BaseEstimator):
    def __init__(self):
        self.activity_classifier = xgb.XGBClassifier(n_estimators=150, max_depth=10)
        self.user_classifier = RandomForestClassifier()
        self.CREATED_BY = "Enias Cailliau"
        self.MODEL_NAME = "Simple Two Staged Classifier"

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
        X_real = X[:, :-1]
        activities = self.activity_classifier.predict(X_real)
        X_with_activity = np.append(X_real, activities.reshape(-1, 1), 1)
        user_probabilities = self.user_classifier.predict_proba(X_with_activity)
        return user_probabilities

    def get_description(self):
        return "{} created by {}".format(self.MODEL_NAME, self.CREATED_BY)


class twinzy(BaseEstimator):
    def __init__(self):
        self.activity_classifier = xgb.XGBClassifier(n_estimators=150, max_depth=10)
        self.user_classifier = random_forest.RF()
        self.CREATED_BY = "Enias Cailliau"
        self.MODEL_NAME = "Two Staged Classifier"

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

    def get_description(self):
        return "{} created by {}".format(self.MODEL_NAME, self.CREATED_BY)
