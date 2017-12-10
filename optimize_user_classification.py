import datetime
import os
import sys
import subprocess

from datetime import datetime
import xgboost as xgb
from GPyOpt.methods import BayesianOptimization
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score

import trainer as t
from model_evaluation import visualiser, CustomKFold
from utils import pandaman
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from utils import handyman


def print_stats(test_features, train_activity_labels, train_features, train_session_id, train_subject_labels):
    pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                         train_subject_labels=train_subject_labels, train_session_id=train_session_id,
                         test_features=test_features)


def plot_curves(estimator, results_location, train_labels, train_features, train_session_id):
    visualiser.plot_learning_curves(estimator, train_features, train_labels, train_session_id,
                                    results_location)
    visualiser.plot_confusion_matrix(estimator, train_features, train_labels, train_session_id, results_location)


yolo = [{'name': 'n_estimators', 'type': 'discrete', 'domain': (500, 5000, 1)},
        {'name': 'max_depth', 'type': 'discrete', 'domain': (2, 10, 1)},
        {'name': 'min_child_weight', 'type': 'discrete', 'domain': (1, 8, 1)},
        {'name': 'gamma', 'type': 'continuous', 'domain': (1e-5, 1)},
        {'name': 'subsample', 'type': 'continuous', 'domain': (0.6, 1.0)},
        {'name': 'colsample_bytree', 'type': 'continuous', 'domain': (0.6, 1.0)},
        {'name': 'reg_alpha', 'type': 'continuous', 'domain': (1e-5, 100)},
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 1)},
        {'name': 'n_folds', 'type': 'discrete', 'domain': (2000, 2000, 1)},
        ]

activity_to_index = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '12': 7, '13': 8, '16': 9, '17': 10,
                     '24': 11}


def generate_folds(train_features, train_activity_labels, train_sessions):
    folds = []
    for num, (train_index, test_index) in enumerate(CustomKFold.cv(4, train_sessions)):
        train_X, test_X = train_features.iloc[train_index], train_features.iloc[test_index]
        train_Y, test_Y = train_activity_labels.iloc[train_index], train_activity_labels.iloc[test_index]
        xg_train = xgb.DMatrix(train_X, label=train_Y)
        xg_test = xgb.DMatrix(test_X, label=test_Y)
        xg_valid = xgb.DMatrix(test_X)
        test_Y_onehot = preprocessing.label_binarize(test_Y, np.unique(test_Y))
        folds.append((xg_train, xg_test, xg_valid, test_Y, test_Y_onehot))
    return folds


def evaluate(param, folds, numRounds):
    auc_scores, acc_scores = [], []
    print("I am using {} rounds".format(numRounds))
    for i in range(4):
        xg_train, xg_test, xg_valid, test_Y, test_Y_onehot = folds[i]
        watchlist = [(xg_train, 'train'), (xg_test, 'test')]
        bst = xgb.train(param, xg_train, numRounds, watchlist, early_stopping_rounds=50)
        y_scores = bst.predict(xg_valid, output_margin=False, ntree_limit=0)
        auc = roc_auc_score(test_Y_onehot, y_scores, average='macro')
        print("---- Intermediate score auc: {}".format(auc))
        auc_scores.append(auc)
    return np.mean(auc_scores), np.std(auc_scores)


def xgbCv(x, folds):
    fs = np.zeros((x.shape[0], 1))
    for i, params in enumerate(x):
        dict_params = {}
        dict_params['n_estimators'] = int(params[0])
        dict_params['max_depth'] = int(params[1])
        dict_params['min_child_weight'] = int(params[2])
        dict_params['gamma'] = params[3]
        dict_params['subsample'] = params[4]
        dict_params['colsample_bytree'] = params[5]
        dict_params['reg_alpha'] = params[6]
        dict_params['learning_rate'] = params[7]
        dict_params['objective'] = 'multi:softprob'
        dict_params['num_class'] = 8
        dict_params['silent'] = 1
        dict_params['n_threads'] = 12
        print(dict_params)
        auc_mean, auc_std = evaluate(dict_params, folds, int(params[8]))
        fs[i] = auc_mean
    return fs


def bayesOpt(folds):
    opt = BayesianOptimization(f=partial(xgbCv, folds=folds),
                               domain=yolo,
                               optimize_restarts=5,
                               acquisition_type='EI',
                               acquisition_weight=0.2,
                               maximize=True)

    opt.run_optimization(max_iter=100, eps=0)

    print('opt_Y')
    print(opt.Y)
    print('opt_X')
    print(opt.X)
    print("selected:")
    print(np.argmin(opt.Y))
    params = opt.X[np.argmin(opt.Y)]
    dict_params = {}
    dict_params['n_estimators'] = int(params[0])
    dict_params['max_depth'] = int(params[1])
    dict_params['min_child_weight'] = int(params[2])
    dict_params['gamma'] = params[3]
    dict_params['subsample'] = params[4]
    dict_params['colsample_bytree'] = params[5]
    dict_params['reg_alpha'] = params[6]
    dict_params['learning_rate'] = params[7]
    dict_params['objective'] = 'multi:softprob'
    dict_params['num_class'] = 8
    dict_params['silent'] = 1
    dict_params['nthread'] = 12
    print("best params")
    print(dict_params)

    auc_mean, auc_std = evaluate(dict_params, folds, int(params[8]))
    print("I have auc: {} +- {}".format(auc_mean, auc_std))


def main():

    base_options = ["ec", "final", "xgboost"]

    options = base_options

    results_location = handyman.calculate_path_from_options("Results", options)


    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    tee = subprocess.Popen(["tee", "final_enias_optimisation.txt"], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())

    # init trainer
    trainer = t.Trainer("")
    # load data from feature file
    _, train_activity_labels, train_subject_labels, train_sessions, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/augmented.pkl'), final=False)
    train_subject_labels = train_subject_labels.apply(lambda x: x - 1)

    train_features = handyman.load_pickle(results_location + "reduced_feature.pkl")

    print_stats(test_features, train_activity_labels, train_features, train_sessions, train_subject_labels)
    global_start_time = datetime.now()

    folds = generate_folds(train_features, train_subject_labels, train_sessions)

    print("starting with bayesian optimisation:")
    bayesOpt(folds)

    time_elapsed = datetime.now() - global_start_time

    print('Time elpased (hh:mm:ss.ms) {}'.format(time_elapsed))


if __name__ == '__main__':
    main()
