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
    options = ["JVH", "activity", "XGB", "LDAwrapped"]
    results_location = os.path.join("Results", '/'.join(options) + "/")
    # init trainer
    trainer = t.Trainer("")
    # load data from feature file
    train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/augmented.pkl'), final=False)

    params = {'reg_alpha': 7.7928156930661805, 'colsample_bytree': 0.69675743513153376, 'silent': 1, 'learning_rate': 0.221324230031307, 'min_child_weight': 1, 'n_estimators': 2000, 'subsample': 0.71026368052012712, 'objective': 'multi:softprob', 'num_class': 12, 'max_depth': 15, 'gamma': 0.58096777615427075, 'nthread' : 8}

    estimator = xgb.XGBClassifier(**params)
    estimator = LDA_wrapper.LDAWrapper(estimator)
    print "----------------- TESTING -----------------"
    # Create a submission
    start = time.time()
    auc_mean, auc_std = trainer.evaluate(estimator, train_features, train_activity_labels, train_session_id, accuracy=True)

    #plot_curves(estimator,results_location,train_activity_labels,train_features,train_session_id)

    estimator.fit(train_features, train_activity_labels)
    local_options = ["XGB", "activity", "LDAWrapped"]
    trainer.save_estimator(estimator, results_location)
    end = time.time()
    print(str(end - start) + "s elapsed")

if __name__ == '__main__':
    main()
