import datetime
import os
from datetime import datetime
import xgboost as xgb
from GPyOpt.methods import BayesianOptimization
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

import trainer as t
from model_evaluation import visualiser, CustomKFold
from utils import pandaman
import numpy as np
from functools import partial
import matplotlib.pyplot as plt


def print_stats(test_features, train_activity_labels, train_features, train_session_id, train_subject_labels):
    pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                         train_subject_labels=train_subject_labels, train_session_id=train_session_id,
                         test_features=test_features)


def plot_curves(estimator, results_location, train_labels, train_features, train_session_id):
    visualiser.plot_learning_curves(estimator, train_features, train_labels, train_session_id,
                                    results_location)
    visualiser.plot_confusion_matrix(estimator, train_features, train_labels, train_session_id, results_location)


yolo = [{'name': 'n_estimators', 'type': 'discrete', 'domain': (500, 5000, 1)},
        {'name': 'min_samples_leaf', 'type': 'discrete', 'domain': (2, 100, 1)},
        ]


def generate_folds(train_features, train_activity_labels, train_sessions):
    folds = []
    for num, (train_index, test_index) in enumerate(CustomKFold.cv(4, train_sessions)):
        train_X, test_X = train_features.iloc[train_index], train_features.iloc[test_index]
        train_Y, test_Y = train_activity_labels.iloc[train_index], train_activity_labels.iloc[test_index]
        test_Y_onehot = preprocessing.label_binarize(test_Y, np.unique(test_Y))
        folds.append((train_X, train_Y, test_X, test_Y, test_Y_onehot))
    return folds


def evaluate(param, folds):
    auc_scores = []
    estimator = RandomForestClassifier(**param)

    for i in range(4):
        train_X, train_Y, test_X, test_Y, test_Y_onehot = folds[i]
        estimator.fit(train_X, train_Y)
        y_scores = estimator.predict_proba(test_X)
        auc = roc_auc_score(test_Y_onehot, y_scores, average='macro')
        print("---- Intermediate AuC score: {}".format(auc))
        auc_scores.append(auc)
    return np.mean(auc_scores), np.std(auc_scores)


def derive_paramset_from_list(params):
    dict_params = {}
    dict_params['n_estimators'] = int(params[0])
    dict_params['min_samples_leaf'] = int(params[0])
    dict_params['oob_score'] = True
    dict_params['n_jobs'] = 8
    return dict_params


def forestCV(x, folds):
    fs = np.zeros((x.shape[0], 1))
    for i, params in enumerate(x):
        dict_params = derive_paramset_from_list(params)
        print(dict_params)
        auc_mean, auc_std = evaluate(dict_params, folds)
        fs[i] = auc_mean
    return fs


def bayesOpt(folds):
    opt = BayesianOptimization(f=partial(forestCV, folds=folds),
                               domain=yolo,
                               optimize_restarts=5,
                               acquisition_type='LCB',
                               acquisition_weight=0.1,
                               maximize=True)

    opt.run_optimization(max_iter=100, eps=0)

    print('opt_Y')
    print(opt.Y)
    print('opt_X')
    print(opt.X)
    print("selected:")
    print(np.argmin(opt.Y))
    params = opt.X[np.argmin(opt.Y)]
    dict_params = derive_paramset_from_list(params)
    print("best params")
    print(dict_params)

    auc_mean, auc_std = evaluate(dict_params, folds)
    print("I have auc: {} +- {}".format(auc_mean, auc_std))

    opt.plot_acquisition("acquisition.png")
    opt.plot_convergence("convergence.png")


def main():
    # try:
    options = ["JVH", "USER", "XGB", "augmented"]
    results_location = os.path.join("Results", '/'.join(options) + "/")

    trainer = t.Trainer()
    train_features, train_activity_labels, train_subject_labels, train_sessions, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/augmented.pkl'), final=False)

    print_stats(test_features, train_activity_labels, train_features, train_sessions, train_subject_labels)
    global_start_time = datetime.now()

    folds = generate_folds(train_features, train_subject_labels, train_sessions)

    print("starting with bayesian optimisation:")
    bayesOpt(folds)

    time_elapsed = datetime.now() - global_start_time

    print('Time elpased (hh:mm:ss.ms) {}'.format(time_elapsed))
    send_notification("Finished", 'Time elpased (hh:mm:ss.ms) {}'.format(time_elapsed))

    os.system("say Your program has finished")
    os.system("say Your program has finished")
    os.system("say Your program has finished")


# except Exception as e:
#     print(e)
#     send_notification("Exception occurred", "{}".format(e))


if __name__ == '__main__':
    main()
