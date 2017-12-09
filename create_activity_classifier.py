import os

import numpy as np

import trainer as t
from model_evaluation import visualiser
from sklearn import ensemble
from utils import pandaman
import xgboost as xgb
import time
from models import LDA_wrapper


def print_stats(test_features, train_activity_labels, train_features, train_session_id, train_subject_labels):
    pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                         train_subject_labels=train_subject_labels, train_session_id=train_session_id,
                         test_features=test_features)


def plot_curves(estimator, results_location, train_labels, train_features, train_session_id):
    visualiser.plot_learning_curves(estimator, train_features, train_labels, train_session_id,
                                    results_location)
    visualiser.plot_confusion_matrix(estimator, train_features, train_labels, train_session_id, results_location)


def evaluate(estimator, train_activity_labels, train_features, train_session_id, trainer):
    auc_mean, auc_std = trainer.evaluate(estimator, train_features, train_activity_labels,
                                                            train_session_id)
    acc_mean, acc_std = trainer.evaluate(estimator, train_features, train_activity_labels,
                                                            train_session_id, accuracy=True)

def main():
    options = ["JVH", "activity", "XGB", "Drek"]
    results_location = os.path.join("Results", '/'.join(options) + "/")
    # init trainer
    trainer = t.Trainer("")
    # load data from feature file

    # BEST
    params = {
        'colsample_bytree': 0.55,
        'silent': 1,
        'learning_rate': 0.10,
        'min_child_weight': 1,
        'n_estimators': 300,
        'subsample': 0.65,
        'objective': 'multi:softprob',
        'max_depth': 5,
        'nthread' : 8,
    }

    # CURRENT
    params = {
        'colsample_bytree': 0.55,
        'silent': 1,
        'learning_rate': 0.10,
        'min_child_weight': 1,
        'n_estimators': 300,
        'subsample': 0.55,
        'objective': 'multi:softprob',
        'max_depth': 5,
        'nthread' : 8,
    }

    estimator = xgb.XGBClassifier(**params)
    print "----------------- TESTING -----------------"
    # Create a submission
    start = time.time()

    train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/unreduced_with_bins.pkl'), final=False)

    auc_mean, auc_std = trainer.evaluate(estimator, train_features, train_activity_labels, train_session_id, accuracy=False)
    if (auc_mean < .9932):
        end = time.time()
        print(str(end - start) + "s elapsed")
        return
    train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/augmented.pkl'), final=False)

    estimator.fit(train_features, train_activity_labels)
    trainer.save_estimator(estimator, results_location)
    end = time.time()
    print(str(end - start) + "s elapsed")

if __name__ == '__main__':
    main()
