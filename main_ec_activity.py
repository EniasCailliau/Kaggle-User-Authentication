import os

import numpy as np
from sklearn import metrics, preprocessing

from sklearn.model_selection import train_test_split

import trainer
from model_evaluation import scorer
from models.activity_prediction import svc
from utils import pandaman, handyman
from feature_reduction import feature_reducer
from model_evaluation import visualiser

options = ["user_logistic_regression", "test"]

results_location = os.path.join("Results", '/'.join(options) + "/")

trainer = trainer.Trainer()

estimator = svc.SVC()

train_features, train_activity_labels, train_subject_labels, test_features = trainer.load_data(
    os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)

pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                     train_subject_labels=train_subject_labels, test_features=test_features)

trainer.evaluate(estimator, train_features, train_activity_labels, scorer.accuracy_evaluator, results_location)

visualiser.plot_learning_curves(estimator, train_features, train_activity_labels, results_location)
#
# X_train, X_test, y_train, y_test = train_test_split(train_features, train_subject_labels, test_size=0.4)
# estimator.fit(X_train, y_train)
# y_pred = estimator.predict_proba(X_test)

# print 'Activity ROC: ' + str(metrics.roc_auc_score(preprocessing.label_binarize(y_test, np.unique(train_activity_labels)), y_pred))
