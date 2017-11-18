import numpy as np
from sklearn import preprocessing
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import roc_auc_score


def auc_evaluator(estimator, X, y):
    y_scores = estimator.predict_proba(X)
    y_one_hot = preprocessing.label_binarize(y, np.unique(y))
    return roc_auc_score(y_one_hot, y_scores, average='macro')


def logloss_evaluator(estimator, X, y):
    y_scores = estimator.predict_proba(X)
    y_one_hot = preprocessing.label_binarize(y, np.unique(y))
    return log_loss(y_one_hot, y_scores)


def accuracy_evaluator(estimator, X, y):
    y_pred = estimator.predict(X)
    # y_one_hot = preprocessing.label_binarize(y, np.unique(y))
    return accuracy_score(y, y_pred)
