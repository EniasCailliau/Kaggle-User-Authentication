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
    options = ["JVH", "XGB", "augmented", "150-10-eta01"]
    results_location = os.path.join("Results", '/'.join(options) + "/")
    # init trainer
    trainer = t.Trainer("")
    # load data from feature file
    train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/augmented.pkl'), final=False)


    params = {
        'nrounds' : 100000,
        'n_estimators' : 150,
        'early.stop.round' : 10,
        'eta' : 0.05,
        'max_depth' : 10,
        'subsample' : 0.70,
        'colsample_bytree' : 0.7,
        'gamma' : 0.1,
        'tree_method' : 'gpu_exact',
        'silent' : 0
    }
    estimator = xgb.XGBClassifier(**params)

    # Create a submission
    start = time.time()
    print "Start"
    auc_mean, auc_std = trainer.evaluate(estimator, train_features, train_subject_labels, train_session_id)
    end = time.time()
    print(str(end - start) + "s elapsed")
    estimator.fit(train_features, train_subject_labels)
    local_options = ["XGB", "aug", "10", str(auc_mean).replace(".","_")]
    trainer.prepare_submission(estimator, test_features, local_options)
    trainer.save_estimator(estimator, results_location)


if __name__ == '__main__':
    main()
