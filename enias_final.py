import csv
import os
import subprocess
import sys
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import xgboost

import trainer as t
from model_evaluation import visualiser
from utils import pandaman, handyman

"""
TODO: write own RFE reduction

"""


def print_stats(test_features, train_activity_labels, train_features, train_session_id, train_subject_labels):
    pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                         train_subject_labels=train_subject_labels, train_session_id=train_session_id,
                         test_features=test_features)


def plot_curves(estimator, results_location, train_labels, train_features, train_session_id):
    # visualiser.plot_learning_curves(estimator, train_features, train_labels, train_session_id,
    #                                 results_location)
    visualiser.plot_auc_learning_curve(estimator, train_features, train_labels, train_session_id,
                                       results_location)
    # visualiser.plot_ce_learning_curve(estimator, train_features, train_labels, train_session_id,
    #                                   results_location)
    # visualiser.plot_confusion_matrix(estimator, train_features, train_labels, train_session_id, results_location)


def main():
    base_options = ["ec", "final", "xgboost"]

    options = base_options + ["FIMP"] + ["semi-optimized"]

    results_location = handyman.calculate_path_from_options("Results", options)
    print("location: {}".format(results_location))

    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    tee = subprocess.Popen(["tee", "final_ec_xgboost.txt"], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())

    trainer = t.Trainer()
    train_features, train_activity_labels, train_subject_labels, train_sessions, test_features = trainer.load_data(
        os.path.join("../feature_extraction", '_data_sets/augmented.pkl'), final=False)

    print_stats(test_features, train_activity_labels, train_features, train_sessions, train_subject_labels)

    """
        Initialize semi optimized estimator
    """
    params = {
        'colsample_bytree': 0.55,
        'silent': 1,
        'learning_rate': 0.10,
        'min_child_weight': 1,
        'n_estimators': 500,
        'subsample': 0.65,
        'objective': 'multi:softprob',
        'max_depth': 5,
        'nthread': 12,
    }
    estimator = xgboost.XGBClassifier(**params)
    #
    print("Fitting estimator...")
    estimator.fit(train_features, train_subject_labels)

    print("Saving estimator...")
    handyman.dump_pickle(estimator, results_location + "estimator.pkl")

    importances = estimator.feature_importances_
    features_ordered = np.argsort(importances)[::-1]

    print("We try on 150 features")
    train_features_reduced = train_features.iloc[:, features_ordered[:150]]

    estimator = handyman.dump_pickle(train_features_reduced, "reduced_feature.pkl")

    print("Tree produced the following importance: ")
    print(estimator.feature_importances_)


if __name__ == '__main__':
    main()
