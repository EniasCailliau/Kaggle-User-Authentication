import os

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from feature_extraction import extractor
from model_evaluation import scorer, CustomKFold
from model_evaluation import visualiser
from utils import create_submission, handyman


class Trainer:
    def __init__(self, rebalance_method=""):
        if rebalance_method == "SMOTEEN":
            print("Trainer initialised with SMOTEEN rebalance method")
            self.rebalancer = SMOTEENN()
        if rebalance_method == "SMOTE":
            print("Trainer initialised with SMOTE rebalance method")
            self.rebalancer = SMOTE()
        else:
            print("Trainer initialised with NO rebalance method")
            self.rebalancer = None

    def find_optimized_model(self, estimator, X, y, train_sessions, tuned_parameters, scorer):

        X_train, X_test, y_train, y_test, sess_train, sess_test = train_test_split(X, y, train_sessions, test_size=0.25,
                                                                                   shuffle=False)
        num_folds = 4

        print("Performing grid search to find best parameter set")
        clf = GridSearchCV(estimator, param_grid=tuned_parameters, scoring=scorer,
                           cv=CustomKFold.cv(num_folds, sess_train),
                           verbose=2, n_jobs=-1)

        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:\n")
        print(clf.best_params_)
        print("Grid scores on development set:\n")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r\n"
                  % (mean, std * 2, params))

        print("Detailed classification report:\n")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.\n")
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        return clf.best_estimator_

    def prepare_submission(self, model, test_data, options):
        """
        :param model: Trained model
        :param test_data: Test data to perform validation on
        :param options: Handy for printout
        :return:
        """
        predictions_test = model.predict_proba(test_data)
        location = os.path.join('Predictions', 'linear' + '-' + '-'.join(options))
        pred_file_name = handyman.generate_unqiue_file_name(
            location, 'npy')
        handyman.dump_npy(predictions_test, pred_file_name)
        print('Dumped predictions to {}'.format(pred_file_name))
        create_submission.main(pred_file_name, pred_file_name + '.csv')

    def __cross_validate(self, estimator, train_features, train_labels, train_session, scorer):
        scores = []
        train_labels = np.array(train_labels.values).ravel()
        train_features = np.array(train_features.values)
        num_folds = 4

        for num, (train_index, test_index) in enumerate(CustomKFold.cv(num_folds, train_session)):
            X_train, X_test = train_features[train_index], train_features[test_index]
            y_train, y_test = train_labels[train_index], train_labels[test_index]
            if self.rebalancer:
                train_features, train_labels = self.rebalancer.fit_sample(train_features, train_labels)
            X_train_rebalanced, y_train_rebalanced = X_train, y_train
            estimator.fit(X_train_rebalanced, y_train_rebalanced)
            score = scorer(estimator, X_test, y_test)
            print("---- Intermediate score: {}".format(score))
            scores.append(score)
        return np.array(scores)

    def evaluate(self, estimator, train_data, train_labels, train_sessions, accuracy=False):
        if accuracy:
            values = self.__cross_validate(estimator, train_data, train_labels, train_sessions,
                                           scorer.accuracy_evaluator)
            print("Accuracy: %0.2f (+/- %0.2f)" % (values.mean(), values.std() * 2))
        else:
            values = self.__cross_validate(estimator, train_data, train_labels, train_sessions, scorer.auc_evaluator)
            print("AuC: %0.2f (+/- %0.2f)" % (values.mean(), values.std() * 2))

        return [values.mean(), values.std()]

    def __rebalance_data(self, X, y):
        if self.rebalancer:
            self.rebalancer.fit_sample(X, y)
        else:
            return [X, y]

    def load_data(self, filepath, final):
        train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = extractor.load_prepared_data_set(
            filepath)

        if self.rebalancer and final:
            print("Trainer is REBALANCING data")
            train_features, train_subject_labels = self.rebalancer.fit_sample(train_features, train_subject_labels)
        else:
            print("Trainer does NOT rebalance _data")
        # Scale the data according to training set
        scaler = StandardScaler()
        train_features_np = scaler.fit_transform(train_features)
        test_features_np = scaler.transform(test_features)
        train_features = pd.DataFrame(train_features_np, columns=train_features.columns.values)
        test_features = pd.DataFrame(test_features_np, columns=test_features.columns.values)
        return train_features, train_activity_labels, train_subject_labels, train_session_id, test_features

    def save_estimator(self, estimator, results_location):
        handyman.dump_pickle(estimator, results_location + "estimator.pkl")

    def load_estimator(self, results_location, verbose=0):
        estimator = handyman.load_pickle(results_location + "estimator.pkl")
        if verbose:
            print("Description from loaded model {}".format(estimator.get_description()))
            print("Parameters from loaded model {}".format(estimator.get_params(deep=True)))
        return estimator
