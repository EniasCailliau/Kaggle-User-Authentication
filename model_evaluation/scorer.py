from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from utils import handyman

def auc_evaluator(estimator, X, y):
    y_scores = estimator.predict_proba(X)
    y_one_hot = handyman.to_one_hot([int(x) for x in y], min_int=1, max_int=8)
    return roc_auc_score(y_one_hot, y_scores, average='macro')


def logloss_evaluator(estimator, X, y):
    y_scores = estimator.predict_proba(X)
    y_one_hot = handyman.to_one_hot([int(x) for x in y], min_int=1, max_int=8)
    return log_loss(y_one_hot, y_scores)



