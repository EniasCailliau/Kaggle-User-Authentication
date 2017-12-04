import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import sys
import xgboost
from sklearn.ensemble import RandomForestClassifier
import subprocess
import trainer as t
from model_evaluation import visualiser, scorer, CustomKFold
from utils import pandaman, handyman
from feature_reduction import feature_analyser
from sklearn.utils import shuffle

def print_stats(test_features, train_activity_labels, train_features, train_session_id, train_subject_labels):
    pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                         train_subject_labels=train_subject_labels, train_session_id=train_session_id,
                         test_features=test_features)


def plot_curves(estimator, results_location, train_labels, train_features, train_session_id):
    visualiser.plot_learning_curves(estimator, train_features, train_labels, train_session_id,
                                    results_location)
    visualiser.plot_confusion_matrix(estimator, train_features, train_labels, train_session_id, results_location)


def visualize_importances(ranking, location):
    ranking = ranking.reshape((5, -1))
    plt.matshow(ranking, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title("Feature importance (forest)")
    plt.savefig(os.path.join(location, "RFE_ranking"), bbox_inches='tight', dpi=300)


def visualize_rfe_importance(rfe, train_features, location):
    importances = rfe.ranking_
    indices = np.argsort(importances)

    with open(location + 'forest_importance.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["index", "feature", "feature name", "importance"])
        print("Feature ranking:")
        n_features = len(importances)
        for f in range(n_features):
            feature_id = indices[f]
            feature_name = train_features.columns.values[feature_id]
            print(
                "%d. feature %d (%s) =  %f" % (f + 1, feature_id, feature_name, importances[feature_id]))
            csv_writer.writerow([f + 1, feature_id, feature_name, importances[feature_id]])

        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importance (rfe)")
        plt.bar(range(n_features), importances[indices],
                color="r", align="center")
        plt.xticks(range(n_features), indices)
        plt.xlim([-1, n_features])
        plt.savefig(location + "forest_importance", bbox_inches='tight', dpi=300)


def visualize_feature_importance(trainer, rfe, estimator, train_features, train_labels, train_sessions, location,
                                 std=False,
                                 ):
    numbers_of_features = []
    means = []
    stds = []
    n_best = 0
    auc_best = 0
    features_ordered = rfe.ranking_.argsort()
    with open(location + 'feature_importance_rfe.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["num_features", "auc_mean", "auc_std"])
        print("going to check for features up to: ", train_features.shape[1] / 2)
        for i in range(1, train_features.shape[1] / 2, 5):
            print("-- Checking performance when {} features are used".format(i))
            train_features_reduced = train_features.iloc[:, features_ordered[:i]]
            if not std:
                (train_index, test_index) = CustomKFold.cv(4, train_sessions)[0]
                X_train, X_test = train_features_reduced.iloc[train_index], train_features_reduced.iloc[test_index]
                y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]
                estimator.fit(X_train, y_train)
                auc_mean = scorer.auc_evaluator(estimator, X_test, y_test)
                auc_std = 0
            else:
                auc_mean, auc_std = trainer.evaluate(estimator, train_features_reduced, train_labels, train_sessions,
                                                     accuracy=False)

            if (i % 100 == 1):
                print("Plotting curves")
                plot_curves(estimator, location + str(i) + "_", train_labels, train_features_reduced, train_sessions)
            if (auc_best < auc_mean):
                print("I upgrade from {} to {} features".format(n_best, i))
                print("because of auc improvement from {} to {}".format(auc_best, auc_mean))
                auc_best = auc_mean
                n_best = i

            numbers_of_features.append(i)
            means.append(auc_mean)
            stds.append(auc_std)
            csv_writer.writerow([i, auc_mean, auc_std])

        plt.figure()
        plt.errorbar(numbers_of_features, means, yerr=stds, marker='o')
        plt.title("Feature importance")
        plt.xlabel("# features")
        plt.ylabel("AuC")
        plt.savefig(location + "feature_importance_rfe", bbox_inches='tight', dpi=300)
        print("Best number of features: {} with auc: {}".format(n_best, auc_best))


def visualize_feature_lda_pca(rfe, train_features, train_labels, location):
    features_ordered = rfe.ranking_.argsort()
    for i in range(1, train_features.shape[1] / 2):
        if i in [10, 20, 30, 40] or (i % 50 == 0):
            print("Plotting curves")
            train_features_reduced = train_features.iloc[:, features_ordered[:i]]
            # print("reducing PCA")
            # feature_analyser.visualise_features_PCA(train_features_reduced, train_labels,
            #                                         os.path.join(location, "PCA" + str(i)))
            print("reducing LDA")
            feature_analyser.visualise_features_LDA(train_features_reduced, train_labels,
                                                    os.path.join(location, "LDA" + str(i)))


def main():
    base_options = ["ec", "feature_analysis", "rfe_augmented"]

    options = base_options

    results_location = handyman.calculate_path_from_options("Results", options)
    print("location: {}".format(results_location))


    trainer = t.Trainer()
    train_features, train_activity_labels, train_subject_labels, train_session, test_features = trainer.load_data(
        os.path.join("../feature_extraction/_data_sets/augmented.pkl"), final=False)
    train_features["activity"] = train_activity_labels


    print_stats(test_features, train_activity_labels, train_features, train_session, train_subject_labels)
    #
    # """
    #     Initialize semi optimized estimator
    # """
    # estimator = RandomForestClassifier(n_estimators=250, criterion='gini', min_samples_leaf=10, oob_score=True,
    #                                    n_jobs=-1)
    #
    # """
    #     Start RFE reduction (stop at 50 features)
    # """
    # train_data_reduced, ranking, rfe = reducer.reduce_RFE(train_features, train_activity_labels, estimator,
    #                                                       n_features_to_select=985)
    #
    # print("Saving RFE...")
    # handyman.dump_pickle(rfe, results_location + "rfe.pkl")

    ranker = handyman.load_pickle(
        "../Results/LS/user_with_activities/random_forest/RFE/augmented/semi-optimizedrfe.pkl")
    ranking = ranker.ranking_
    print("RFE produced the following ranking: ")
    print(ranking)


    # train_features, train_subject_labels = shuffle(train_features, train_subject_labels)

    # train_features = train_features.iloc[:8000].reset_index(drop=True)
    # train_activity_labels = train_activity_labels.iloc[:8000].reset_index(drop=True)
    # train_subject_labels = train_subject_labels.iloc[:8000].reset_index(drop=True)
    # train_session = train_subject_labels.iloc[:8000].reset_index(drop=True)

    print_stats(test_features, train_activity_labels, train_features, train_session, train_subject_labels)

    handyman.dump_pickle(ranker, results_location + "/ranker.pkl")

    # visualize_importances(ranking, results_location)ze!
    # visualize_rfe_importance(ranker, train_features, results_location)

    visualize_feature_lda_pca(ranker, train_features, train_subject_labels, results_location)
    os.system('say Your program has finished!')
    os.system('say Your program has finished!')
    os.system('say Your program has finished!')


if __name__ == '__main__':
    main()
