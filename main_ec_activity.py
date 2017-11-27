from os import system

import trainer as t
from feature_reduction import feature_reducer as reducer
from feature_reduction.feature_reducer import Scorer
from model_evaluation import visualiser, scorer
from models.activity_prediction import xgboost_activity, random_forest_activity
from utils import pandaman, handyman
import matplotlib.pyplot as plt
import os


def print_stats(test_features, train_activity_labels, train_features, train_session_id, train_subject_labels):
    pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                         train_subject_labels=train_subject_labels, train_session_id=train_session_id,
                         test_features=test_features)


def plot_curves(estimator, results_location, train_labels, train_features, train_session_id):
    visualiser.plot_learning_curves(estimator, train_features, train_labels, train_session_id,
                                    results_location)
    visualiser.plot_confusion_matrix(estimator, train_features, train_labels, train_session_id, results_location)


def evaluate(estimator, train_activity_labels, train_features, train_session_id, trainer):
    auc_mean, auc_std, acc_mean, acc_std = trainer.evaluate(estimator, train_features, train_activity_labels,
                                                            train_session_id)

    print("AuC: {} \t Acc:{}".format(auc_mean, acc_mean))
    return [auc_mean, auc_std, acc_mean, acc_std]


def main():
    base_options = ["ec", "activity", "random_forest"]

    trainer = t.Trainer()
    train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)

    print_stats(test_features, train_activity_labels, train_features, train_session_id, train_subject_labels)


    train_data_reduced, scores, select_k_best = reducer.reduce_k_best(train_features, train_activity_labels, Scorer.F_CLASSIF)
    print(scores)
    estimator = random_forest_activity.RandomForest(n_estimators=250, criterion='gini', oob_score=True)
    options = base_options + ["reduce300"] + ["gridsearch_attempt2"]

    results_location = handyman.calculate_path_from_options("Results", options)
    print("location: {}".format(results_location))

    # param_grid = [{'n_estimators': [25, 100, 150, 200, 250, 300],
    #                'max_features': ['auto', None],
    #                'min_samples_leaf': [1, 5, 10, 30, 50],
    #                'class_weight': [None, 'balanced']
    #                }]


    # estimator = trainer.find_optimized_model(estimator, train_features, train_activity_labels, train_session_id,
    #                                          param_grid, scorer.auc_evaluator)
    trainer.save_estimator(estimator, results_location)
    auc_mean, auc_std, acc_mean, acc_std = evaluate(estimator, train_activity_labels, train_features,
                                                    train_session_id,
                                                    trainer)
    print("Auc: {}".format(auc_mean))
    print("Acc: {}".format(acc_mean))

    system('say Your program has finished!')
    system('say Your program has finished!')
    system('say Your program has finished!')


if __name__ == '__main__':
    main()
