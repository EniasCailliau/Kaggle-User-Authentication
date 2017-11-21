import os

import numpy as np
from sklearn import metrics, preprocessing

from sklearn.model_selection import train_test_split

import trainer
from feature_reduction.feature_reducer import Scorer
from model_evaluation import scorer
from models.activity_prediction import svc
from utils import pandaman, handyman
from feature_reduction import feature_reducer
from model_evaluation import visualiser

options = ["ec", "activity_svc", "gridsearch"]

results_location = os.path.join("Results", '/'.join(options) + "/")

trainer = trainer.Trainer()

estimator = svc.SVC()

train_features, train_activity_labels, train_subject_labels, test_features = trainer.load_data(
    os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)

pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                     train_subject_labels=train_subject_labels, test_features=test_features)

train_features_reduced, scores, select_k_best = feature_reducer.reduce_k_best(train_features, train_activity_labels,
                                                                              Scorer.F_CLASSIF, k=300)
pandaman.print_stats(train_features_reduced=train_features_reduced)

# trainer.evaluate(estimator, train_features_reduced, train_activity_labels, scorer.accuracy_evaluator, results_location)


param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'shrinking': [True, False],
     'verbose': [True]}
]

estimator = trainer.find_optimized_model(estimator, train_features_reduced, train_subject_labels, param_grid,
                                         scorer.accuracy_evaluator)

trainer.save_estimator(estimator, results_location)
estimator = trainer.load_estimator(results_location)

# visualiser.plot_learning_curves(estimator, train_features, train_activity_labels, results_location)
#
# X_train, X_test, y_train, y_test = train_test_split(train_features, train_subject_labels, test_size=0.4)
# estimator.fit(X_train, y_train)
# y_pred = estimator.predict_proba(X_test)

# print 'Activity ROC: ' + str(metrics.roc_auc_score(preprocessing.label_binarize(y_test, np.unique(train_activity_labels)), y_pred))
