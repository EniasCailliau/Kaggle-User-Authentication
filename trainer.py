import os

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from model_evaluation import scorer

from feature_extraction import extractor
from model_evaluation import visualiser
from model_evaluation import scorer as score_evaluation
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

    def find_optimized_model(self, estimator, X, y, tuned_parameters, scorer):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

        print("Performing grid search to find best parameter set")
        clf = GridSearchCV(estimator, param_grid=tuned_parameters, scoring=scorer, cv=StratifiedKFold(n_splits=4),
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

    def __cross_validate(self, estimator, train_features, train_labels, scorer):
        scores = []
        train_labels = np.array(train_labels.values)
        train_features = np.array(train_features.values)
        skf = list(StratifiedKFold(n_splits=4)
                   .split(train_features, train_labels))
        for num, (train_index, test_index) in enumerate(skf):
            X_train, X_test = train_features[train_index], train_features[test_index]
            y_train, y_test = train_labels[train_index], train_labels[test_index]
            if self.rebalancer:
                train_features, train_labels = self.rebalancer.fit_sample(train_features, train_labels)
            X_train_rebalanced, y_train_rebalanced = X_train, y_train
            estimator.fit(X_train_rebalanced, y_train_rebalanced)

            score = scorer(estimator, X_test, y_test)
            scores.append(score)
        return np.array(scores)

    def evaluate(self, estimator, train_data, train_labels, location):
        print("--------evaluation--------")
        X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.4)
        estimator.fit(X_train, y_train)
        print("For my random training set I have following auc_roc score:")
        print("AuC: {}".format(scorer.auc_evaluator(estimator, X_test, y_test)))
        print("Accuracy: {}".format(scorer.accuracy_evaluator(estimator, X_test, y_test)))

        auc_values = self.__cross_validate(estimator, train_data, train_labels, scorer.auc_evaluator)
        print(auc_values)
        print("AuC: %0.2f (+/- %0.2f)" % (auc_values.mean(), auc_values.std() * 2))
        print("------------------------------------------------------------------------")
        acc_values = self.__cross_validate(estimator, train_data, train_labels, scorer.accuracy_evaluator)
        print(acc_values)
        print("Accuracy: %0.2f (+/- %0.2f)" % (acc_values.mean(), acc_values.std() * 2))
        visualiser.plot_confusion_matrix(estimator, X_test, y_test, location)
        return [auc_values.mean(), acc_values.mean()]

    def get_acc_auc(self, estimator, train_data, train_labels, location):
        print("--------evaluation--------")
        scores = self.__cross_validate(estimator, train_data, train_labels, score_evaluation.accuracy_evaluator)
        accuracy = scores.mean()
        print("Accuracy Score: {}".format(accuracy))
        scores = self.__cross_validate(estimator, train_data, train_labels, score_evaluation.auc_evaluator)
        auc = scores.mean()
        print("AUC_ROC Score: {}".format(auc))

        X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.4)
        estimator.fit(X_train, y_train)
        visualiser.plot_confusion_matrix(estimator, X_test, y_test, location)
        return [accuracy, auc]

    def __rebalance_data(self, X, y):
        if self.rebalancer:
            self.rebalancer.fit_sample(X, y)
        else:
            return [X, y]

    def load_data(self, filepath, final):
        train_features, train_activity_labels, train_subject_labels, test_features = extractor.load_prepared_data_set(
            filepath)
        if self.rebalancer and final:
            print("Trainer is REBALANCING data")
            train_features, train_subject_labels = self.rebalancer.fit_sample(train_features, train_subject_labels)
        else:
            print("Trainer does NOT rebalance _data")
        # Shuffle the data
        train_features, train_activity_labels, train_subject_labels = shuffle(train_features, train_activity_labels,
                                                                              train_subject_labels)
        # Scale the data according to training set
        scaler = StandardScaler()
        train_features_np = scaler.fit_transform(train_features)
        test_features_np = scaler.transform(test_features)
        train_features = pd.DataFrame(train_features_np, columns=train_features.columns.values)
        test_features = pd.DataFrame(test_features_np, columns=test_features.columns.values)
        return train_features, train_activity_labels, train_subject_labels, test_features

    def save_estimator(self, estimator, results_location):
        handyman.dump_pickle(estimator, results_location + "estimator.pkl")

    def load_estimator(self, results_location, verbose=0):
        estimator = handyman.load_pickle(results_location + "estimator.pkl")
        if verbose:
            print("Description from loaded model {}".format(estimator.get_description()))
            print("Parameters from loaded model {}".format(estimator.get_params(deep=True)))
        return estimator
