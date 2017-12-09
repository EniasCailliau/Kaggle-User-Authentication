import csv
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import xgboost

import trainer as t
from model_evaluation import visualiser
from utils import pandaman, handyman

"""
TODO: write own RFE reduction

"""


def print_stats(test_features, train_activity_labels, train_features, train_session_id, train_subject_labels):
    pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                         train_subject_labels=train_subject_labels, train_session_id=train_session_id,
                         test_features=test_features)


def plot_curves(estimator, results_location, train_labels, train_features, train_session_id):
    # visualiser.plot_learning_curves(estimator, train_features, train_labels, train_session_id,
    #                                 results_location)
    visualiser.plot_auc_learning_curve(estimator, train_features, train_labels, train_session_id,
                                       results_location)
    # visualiser.plot_ce_learning_curve(estimator, train_features, train_labels, train_session_id,
    #                                   results_location)
    # visualiser.plot_confusion_matrix(estimator, train_features, train_labels, train_session_id, results_location)


def visualize_importances(importance, location):
    importance = importance / max(importance)

    ranking = importance.reshape((17, -1))
    plt.matshow(ranking, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title("Feature importance (forest)")
    plt.savefig(location + "RFE_ranking", bbox_inches='tight', dpi=300)


def visualize_forest_importance(forest, train_features, location):
    importances = forest.feature_importances_
    importances = importances / max(importances)
    indices = np.argsort(-importances)

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
        plt.title("Feature importance (forest)")
        plt.bar(range(n_features), importances[indices],
                color="r", align="center")
        plt.xticks(range(n_features), indices)
        plt.xlim([-1, n_features])
        plt.savefig(location + "forest_importance", bbox_inches='tight', dpi=300)


def visualize_feature_importance_tree(trainer, forest, train_features, train_labels, train_sessions, location):
    importances = forest.feature_importances_
    features_ordered = np.argsort(importances)[::-1]

    numbers_of_features = []
    means = []
    stds = []
    n_best = 0
    auc_best = 0
    with open(location + 'feature_importance_forest.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["num_features", "auc_mean", "auc_std"])
        print("going to check for features up to: ", len(importances) / 2)
        for i in range(1, len(importances) / 2, 5):
            print("-- Checking performance when {} features are used".format(i))
            train_features_reduced = train_features.iloc[:, features_ordered[:i]]
            auc_mean, auc_std = trainer.evaluate(forest, train_features_reduced, train_labels, train_sessions)
            csv_writer.writerow([i, auc_mean, auc_std])

            if (i % 100 == 1):
                print("Plotting curves")
                plot_curves(forest, location + str(i) + "_", train_labels, train_features_reduced, train_sessions)
            if (auc_best < auc_mean):
                print("I upgrade from {} to {} features".format(n_best, i))
                print("because of auc improvement from {} to {}".format(auc_best, auc_mean))
                auc_best = auc_mean
                n_best = i
            numbers_of_features.append(i)
            means.append(auc_mean)
            stds.append(auc_std)

        plt.figure()
        plt.figure()
        plt.errorbar(numbers_of_features, means, yerr=stds, marker='o')
        plt.title("Feature importance")
        plt.xlabel("# features")
        plt.ylabel("AuC")
        plt.savefig(location + "feature_importance_forest", bbox_inches='tight', dpi=300)
        print("Best number of features: {} with auc: {}".format(n_best, auc_best))


def main():
    base_options = ["ec", "final", "xgboost"]

    options = base_options + ["FIMP"] + ["semi-optimized"]

    results_location = handyman.calculate_path_from_options("Results", options)
    print("location: {}".format(results_location))

    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    tee = subprocess.Popen(["tee", "final_ec_xgboost.txt"], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())

    trainer = t.Trainer()
    train_features, train_activity_labels, train_subject_labels, train_sessions, test_features = trainer.load_data(
        os.path.join("../feature_extraction", '_data_sets/augmented.pkl'), final=False)

    print_stats(test_features, train_activity_labels, train_features, train_sessions, train_subject_labels)

    """
        Initialize semi optimized estimator
    """
    params = {
        'colsample_bytree': 0.55,
        'silent': 1,
        'learning_rate': 0.10,
        'min_child_weight': 1,
        'n_estimators': 500,
        'subsample': 0.65,
        'objective': 'multi:softprob',
        'max_depth': 5,
        'nthread': 12,
    }
    estimator = xgboost.XGBClassifier(**params)
    #
    print("Fitting estimator...")
    estimator.fit(train_features, train_subject_labels)

    print("Saving estimator...")
    handyman.dump_pickle(estimator, results_location + "estimator.pkl")

    # estimator = handyman.load_pickle(results_location + "estimator.pkl")

    print("Tree produced the following importance: ")
    print(estimator.feature_importances_)

    print("Visualising forest importance in grid...")
    visualize_importances(estimator.feature_importances_, results_location)

    print("Visualising forest importance...")
    visualize_forest_importance(estimator, train_features, results_location)

    print("Visualising feature importance...")
    visualize_feature_importance_tree(trainer, estimator, train_features, train_subject_labels, train_sessions,
                                      results_location)

    print("Visualising learning curve...")
    plot_curves(estimator, results_location, train_subject_labels, train_features, train_sessions)


if __name__ == '__main__':
    main()
