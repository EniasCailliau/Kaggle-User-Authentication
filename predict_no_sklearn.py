import datetime
import os
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


def print_stats(test_features, train_activity_labels, train_features, train_session_id, train_subject_labels):
    pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                         train_subject_labels=train_subject_labels, train_session_id=train_session_id,
                         test_features=test_features)


def plot_curves(estimator, results_location, train_labels, train_features, train_session_id):
    visualiser.plot_learning_curves(estimator, train_features, train_labels, train_session_id,
                                    results_location)
    visualiser.plot_confusion_matrix(estimator, train_features, train_labels, train_session_id, results_location)

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


def evaluate(param, folds, numRounds, n_iter=1):
    auc_scores, acc_scores = [], []
    print("I am using {} rounds".format(numRounds))
    for i in range(n_iter):
        xg_train, xg_test, xg_valid, test_Y, test_Y_onehot = folds[i]
        watchlist = [(xg_train, 'train'), (xg_test, 'test')]
        bst = xgb.train(param, xg_train, numRounds, watchlist, early_stopping_rounds=50)
        y_scores = bst.predict(xg_valid, output_margin=False, ntree_limit=0)
        auc = roc_auc_score(test_Y_onehot, y_scores, average='macro')
        print("---- Intermediate score auc: {}".format(auc))
        auc_scores.append(auc)

        y_pred = np.argmax(y_scores, axis=1)
        acc = accuracy_score(test_Y, y_pred)
        print("---- Intermediate score acc: {}".format(acc))
        acc_scores.append(acc)
    return np.mean(auc_scores), np.std(auc_scores), np.mean(acc_scores), np.std(acc_scores)


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
        print(dict_params)
        auc_mean, auc_std, acc_mean, acc_std = evaluate(dict_params, folds, int(params[8]))
        fs[i] = auc_mean
    return fs

def main():
    # try:
    options = ["JVH", "USER", "XGB", "augmented"]
    results_location = os.path.join("Results", '/'.join(options) + "/")
    # init trainer
    trainer = t.Trainer("")
    # load data from feature file
    train_features, train_activity_labels, train_subject_labels, train_sessions, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/augmented.pkl'), final=False)
    train_subject_labels = train_subject_labels.apply(lambda x: x - 1)

    print_stats(test_features, train_activity_labels, train_features, train_sessions, train_subject_labels)
    global_start_time = datetime.now()


    folds = generate_folds(train_features, train_subject_labels, train_sessions)


    print("Starting validation...")
    params = {'reg_alpha': 22.517068955288213, 'colsample_bytree': 0.95241906379281249, 'silent': 1, 'learning_rate': 0.790592831047362,
              'min_child_weight': 8, 'n_estimators': 5000, 'subsample': 0.73530308049441184, 'objective': 'multi:softprob',
              'num_class': 8, 'max_depth': 2, 'gamma': 0.79861290730045043, 'nthread': 8}

    print params

    auc_mean, auc_std, acc_mean, acc_std = evaluate(params, folds, 2000, n_iter=4)
    print("I have auc: {} +- {}".format(auc_mean, auc_std))
    print("I have acc: {} +- {}".format(acc_mean, acc_std))

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
