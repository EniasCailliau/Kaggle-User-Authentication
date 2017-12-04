import os

import numpy as np

import trainer as t
from model_evaluation import visualiser
from models.subject_prediction import random_forest
from sklearn import ensemble
from utils import pandaman
import xgboost as xgb
import time


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
    options = ["JVH", "activity","XGB", "unreduced", "temp"]
    results_location = os.path.join("Results", '/'.join(options) + "/")
    # init trainer
    trainer = t.Trainer("")
    # load data from feature file
    train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/augmented.pkl'), final=False)

    params = {
        'nrounds' : 100000,
        'n_estimators' : 250,
        'early.stop.round' : 50,
        'eta' : 0.1,
        'max_depth' : 4,
        'min_child_weight' : 1,
        'subsample' : .7,
        'colsample_bytree' : .7,
        'gamma' : 0.05,
        'scale_pos_weight' : 1,
        'nthread' : 8,
    }

    estimator = xgb.XGBClassifier(**params)
    print "----------------- TESTING -----------------"
    # Create a submission
    start = time.time()
    #auc_mean, auc_std = trainer.evaluate(estimator, train_features, train_activity_labels, train_session_id)

    #plot_curves(estimator,results_location,train_activity_labels,train_features,train_session_id)

    estimator.fit(train_features, train_subject_labels)
    local_options = ["XGB", "aug", "optimal"]
    #trainer.prepare_submission(estimator, test_features, local_options)
    trainer.save_estimator(estimator, results_location)
    end = time.time()
    print(str(end - start) + "s elapsed")

if __name__ == '__main__':
    main()
