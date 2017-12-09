import os

import numpy as np

import trainer as t
from model_evaluation import visualiser
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
    options = ["JVH", "user", "XGB", "9951"]
    results_location = os.path.join("Results", '/'.join(options) + "/")
    # init trainer
    trainer = t.Trainer("")
    # load data from feature file
    train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/augmented.pkl'), final=False)

    params = {'reg_alpha': 5.421745185295384, 'colsample_bytree': 0.97401524090312686, 'silent': 1,
              'learning_rate': 0.28981503893663102, 'min_child_weight': 1, 'n_estimators': 100,
              'subsample': 0.72047190882139611,
              'objective': 'multi:softprob', 'max_depth': 2, 'gamma': 0.79234113609335632}

    estimator = xgb.XGBClassifier(**params)
    print "----------------- TESTING -----------------"
    # Create a submission
    start = time.time()
    auc_mean, auc_std = trainer.evaluate(estimator, train_features, train_subject_labels, train_session_id,
                                         accuracy=False)
    print("I have auc: {} +- {}".format(auc_mean, auc_std))

    # plot_curves(estimator,results_location,train_activity_labels,train_features,train_session_id)

    estimator.fit(train_features, train_subject_labels)
    local_options = ["XGB", "aug", "optimal_12_07"]
    trainer.prepare_submission(estimator, test_features, local_options)
    trainer.save_estimator(estimator, results_location)
    end = time.time()
    print(str(end - start) + "s elapsed")


if __name__ == '__main__':
    main()
