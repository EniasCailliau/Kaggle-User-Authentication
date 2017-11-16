import trainer
import os
from models import logistic_regression
from model_evaluation import scorer, visualiser
from feature_reduction import feature_reducer
from utils import pandaman, handyman

trainer = trainer.Trainer()

estimator = logistic_regression.LogReg()

train_features, train_activity_labels, train_subject_labels, test_features = trainer.load_data(
    os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)

pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                     train_subject_labels=train_subject_labels, test_features=test_features)

train_features_reduced, scores, select_k_best = feature_reducer.reduce_k_best(train_features, train_subject_labels,
                                                                              feature_reducer.Scorer.F_CLASSIF, k=10)
pandaman.print_stats(train_features_reduced=train_features_reduced)

train_data_reduced, scores, select_percentile = feature_reducer.reduce_percentile(train_features, train_subject_labels,
                                                                                  feature_reducer.Scorer.F_CLASSIF,
                                                                                  percentile=10)
pandaman.print_stats(train_features_reduced=train_features_reduced)

train_data_reduced = feature_reducer.reduce_variance(train_features, p=0.95)
pandaman.print_stats(train_features_reduced=train_features_reduced)

train_data_reduced, importances, forest = feature_reducer.reduce_tree(train_features, train_activity_labels)
feature_reducer.visualize_tree_rankin(forest, 10)
pandaman.print_stats(train_features_reduced=train_features_reduced)

"""
    ATTENTION: RFE takes a long time so it is important to save your progress as a new dataset (see code below)
"""

n_features = 100
estimator_name = "logreg"
train_data_reduced, scores, rfe = feature_reducer.reduce_RFE(train_features, train_activity_labels, estimator,
                                                             n_features_to_select=100)
feature_reducer.visualize_RFE_ranking(rfe)
pandaman.print_stats(train_features_reduced=train_features_reduced)

print(rfe.ranking_)

train_features_new = rfe.transform(train_features)
test_features_new = rfe.transform(test_features)
handyman.dump_pickle(
    dict(train_features=train_features_new, train_activity_labels=train_activity_labels,
         train_subject_labels=train_subject_labels, test_features=test_features_new, ranking=rfe.ranking_),
    os.path.join("feature_extraction", "_data_sets/rfe_" + estimator_name + "_" + str(n_features) + ".pkl")
)
