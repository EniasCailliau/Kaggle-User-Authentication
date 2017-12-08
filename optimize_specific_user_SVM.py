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
from sklearn import cross_decomposition, naive_bayes, neural_network, svm



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


    for x in ['1','2','3','5','6','12','13','16','17','24']:
        # load data from feature file
        train_features, train_activity_labels, train_subject_labels, train_sessions, test_features = trainer.load_data(
            os.path.join("feature_extraction", '_data_sets/augmented.pkl'), final=False)
        index = train_activity_labels == x

        train_features =  train_features[index].reset_index(drop=True)
        train_sessions = train_sessions[index].reset_index(drop=True)
        train_subject_labels = train_subject_labels[index].reset_index(drop=True)
        train_activity_labels = train_activity_labels.reset_index(drop=True)

        print_stats(test_features, train_activity_labels, train_features, train_sessions, train_subject_labels)

        options = ["JVH", "SVM", "user", str(x)]
        results_location = os.path.join("Results", '/'.join(options) + "/")
        print "----------------------------------------------"
        print "Start " + str(x)
        start = time.time()
        current_best_score = 0;
        current_best_params = {}

        for iteration in range(25):
            print "-- ITERATION " + str(iteration) + " --"
            params = {}
            params['C'] = math.pow(0.1, 4*np.random.random()-2)
            params['probability'] = True
            params['tol'] = 1e-4
            params['class_weight'] = 'balanced'
            params['random_state'] = np.random.random()

            print params
            estimator = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
            auc_mean, auc_std = trainer.evaluate(estimator, train_features, train_subject_labels, train_sessions)
            if(auc_mean > current_best_score):
                print "############################## NEW BEST: " + str(auc_mean)
                current_best_score = auc_mean
                current_best_params = current_best_params
                estimator.fit(train_features, train_subject_labels)
                trainer.save_estimator(estimator, results_location, filename="MLP_"+str(x)+"_"+str(int(auc_mean*100000))+".pkl")
                handyman.dump_pickle(current_best_params, results_location+"MLP_"+str(x)+"_"+str(int(auc_mean*100000))+"_params.pkl")

        end = time.time()
        print(str(end - start) + "s elapsed")

if __name__ == '__main__':
    main()
