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


    # ['1','2','3','5','6','12','13','16','17','24']
    for x in ['16','17']:
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

        for iteration in range(10):
            print "-- ITERATION " + str(iteration) + " --"
            params = {}
            num_layers = np.random.randint(0,4)
            params['hidden_layer_sizes'] = (np.random.randint(75,150),)
            for i in range(num_layers):
                params['hidden_layer_sizes'] = params['hidden_layer_sizes'] + (np.random.randint(75,150),)
            params['alpha'] = math.pow(0.1, 3 + 2*np.random.random())
            params['learning_rate_init'] = math.pow(0.1, 2 + 2*np.random.random())
            params['max_iter'] = 500
            params['random_state'] = np.random.randint(1000000)
            params['tol'] = 0.0000001
            params['momentum'] = 0.8 + 0.2*np.random.random()
            params['beta_1'] = 0.8 + 0.2*np.random.random()
            params['beta_2'] = 0.998 + 0.002*np.random.random()
            params['epsilon'] = math.pow(0.1, 7 + 2*np.random.random())
            print params
            estimator = neural_network.MLPClassifier(**params)
            estimator = LDA_wrapper.LDAWrapper(estimator)

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
