from os import system

import trainer as t
from feature_reduction import feature_reducer as reducer
from feature_reduction.feature_reducer import Scorer
from model_evaluation import visualiser, scorer
from models.activity_prediction import xgboost_activity, random_forest_user
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

    auc_means, auc_stds, acc_means, acc_stds, numbers_of_features = [], [], [], [], []

    for k in [10]:
        print("Starting with reduction to k= {}".format(k))
        system('say ' + 'Starting with reduction to k= {}'.format(k))

        estimator = random_forest_user.RandomForest(n_estimators=250, criterion='gini', oob_score=True)
        options = base_options + ["best_" + str(k)] + ["n_estimators=200, criterion=gini, oob_score=True"]

        results_location = handyman.calculate_path_from_options("Results", options)
        print("location: {}".format(results_location))

        # print("K best")
        # train_features_reduced, _, _ = reducer.reduce_k_best(train_features, train_activity_labels, Scorer.MUTUAL_INFO_CLASSIF, k=600)
        print("LDA")
        train_features_reduced, _, _ = reducer.reduce_LDA(train_features, train_activity_labels, n_components=12)
        print("done with lda")
        auc_mean, auc_std, acc_mean, acc_std = evaluate(estimator, train_activity_labels, train_features_reduced,
                                                        train_session_id,
                                                        trainer)
        auc_means.append(auc_mean)
        auc_stds.append(auc_std)
        acc_means.append(acc_mean)
        acc_stds.append(acc_std)
        numbers_of_features.append(k)

    # print("auc_means:")
    # print(auc_means)
    # print("auc_stds:")
    # print(auc_stds)
    # print("acc_means:")
    # print(acc_means)
    # print("acc_stds:")
    # print(acc_stds)
    #
    # plt.figure()
    # plt.errorbar(numbers_of_features, auc_means, yerr=auc_stds, marker='o')
    # plt.title("Feature selection (SelectKBest)")
    # plt.xlabel("# features")
    # plt.ylabel("AuC")
    # plt.savefig("Feature_importance1", bbox_inches='tight', dpi=300)
    #
    # plt.clf()
    # plt.errorbar(numbers_of_features, acc_means, yerr=acc_stds, marker='o')
    # plt.title("Feature selection (selectKBest")
    # plt.xlabel("# features")
    # plt.ylabel("Accuracy")
    # plt.savefig("Feature_importance2", bbox_inches='tight', dpi=300)
    #
    # print("auc_means:")
    # print(auc_means)
    # print("auc_std:")
    # print(auc_stds)
    # print("acc_mean:")
    # print(acc_means)
    # print("acc_std:")
    # print(acc_stds)

    system('say Your program has finished!')
    system('say Your program has finished!')
    system('say Your program has finished!')


if __name__ == '__main__':
    main()
