import numpy as np
from sklearn import preprocessing
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import roc_auc_score


def auc_evaluator(estimator, X, y):
    y_scores = estimator.predict_proba(X)
    y_one_hot = preprocessing.label_binarize(y, np.unique(y))
    return roc_auc_score(y_one_hot, y_scores, average='macro')

def safe_auc_evaluator(estimator, X, y):
    y_scores = estimator.predict_proba(X)
    classes = estimator.classes_
    y_one_hot = preprocessing.label_binarize(y, np.unique(y))
    safe_y_scores = np.zeros(y_one_hot.shape)
    current_column = 0
    for i in range(len(classes)):
        if classes[i] in np.unique(y):
            safe_y_scores[:, current_column] = y_scores[:, i]
            current_column+=1
    return roc_auc_score(y_one_hot, safe_y_scores, average='macro')


def logloss_evaluator(estimator, X, y):
    y_scores = estimator.predict_proba(X)
    y_one_hot = preprocessing.label_binarize(y, np.unique(y))
    return log_loss(y_one_hot, y_scores)

def safe_logloss_evaluator(estimator, X, y):
    y_scores = estimator.predict_proba(X)
    classes = estimator.classes_
    y_one_hot = preprocessing.label_binarize(y, np.unique(y))
    safe_y_scores = np.zeros(y_one_hot.shape)
    current_column = 0
    for i in range(len(classes)):
        if classes[i] in np.unique(y):
            safe_y_scores[:, current_column] = y_scores[:, i]
            current_column+=1
    return log_loss(y_one_hot, safe_y_scores)


def accuracy_evaluator(estimator, X, y):
    y_pred = estimator.predict(X)
    return accuracy_score(y, y_pred)
