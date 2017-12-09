import os
import math
import numpy as np
from utils import handyman
import trainer as t
from model_evaluation import visualiser
from feature_reduction import feature_reducer
from sklearn import ensemble, svm
from utils import pandaman
import xgboost as xgb
import time
import pandas as pd
from sklearn import feature_selection
from sklearn import cross_decomposition, naive_bayes, neural_network
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
    # init trainer
    trainer = t.Trainer("")


    x = '1'
    # load data from feature file
    train_features, train_activity_labels, train_subject_labels, train_sessions, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/augmented.pkl'), final=False)
    index = train_activity_labels == x

    train_features =  train_features[index].reset_index(drop=True)
    train_sessions = train_sessions[index].reset_index(drop=True)
    train_subject_labels = train_subject_labels[index].reset_index(drop=True)
    train_activity_labels = train_activity_labels.reset_index(drop=True)

    print_stats(test_features, train_activity_labels, train_features, train_sessions, train_subject_labels)

    options = ["JVH", "MLP", "user", str(x)]
    results_location = os.path.join("Results", '/'.join(options) + "/")
    print "----------------------------------------------"
    print "Start " + str(x)
    start = time.time()
    current_best_score = 0;
    current_best_params = {}

    params = handyman.load_pickle(results_location+"MLP_1_99398_params.pkl")
    print params
    estimator = neural_network.MLPClassifier(**params)
    estimator = LDA_wrapper.LDAWrapper(estimator)

    auc_mean, auc_std = trainer.evaluate(estimator, train_features, train_subject_labels, train_sessions)

    trainer.save_estimator(estimator, results_location, filename="MLP_WRAPPED_"+str(x)+"_"+str(int(auc_mean*100000))+".pkl")
    handyman.dump_pickle(current_best_params, results_location+"MLP_WRAPPED_"+str(x)+"_"+str(int(auc_mean*100000))+"_params.pkl")

    end = time.time()
    print(str(end - start) + "s elapsed")

if __name__ == '__main__':
    main()
