import csv
import os

import matplotlib.pyplot as plt
import sys
from sklearn.ensemble import RandomForestClassifier
import subprocess

import trainer as t
from feature_reduction import feature_reducer as reducer
from model_evaluation import scorer, CustomKFold, visualiser
from utils import pandaman, handyman


def print_stats(test_features, train_activity_labels, train_features, train_session_id, train_subject_labels):
    pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                         train_subject_labels=train_subject_labels, train_session_id=train_session_id,
                         test_features=test_features)


def plot_curves(estimator, results_location, train_labels, train_features, train_session_id):
    visualiser.plot_learning_curves(estimator, train_features, train_labels, train_session_id,
                                    results_location)
    visualiser.plot_confusion_matrix(estimator, train_features, train_labels, train_session_id, results_location)


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


def main():
    base_options = ["ec", "user_with_activities", "random_forest"]

    options = base_options + ["RFE"] + ["semi-optimized"]

    results_location = handyman.calculate_path_from_options("Results", options)
    print("location: {}".format(results_location))

    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    tee = subprocess.Popen(["tee", "forest_rfe.txt"], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    print "\nstdout"

    trainer = t.Trainer()
    train_features, train_activity_labels, train_subject_labels, train_session, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets', 'unreduced.pkl'), final=False)

    """
        Add activity labels as feature
    """

    train_features["activity"] = train_activity_labels

    print_stats(test_features, train_activity_labels, train_features, train_session, train_subject_labels)

    """
        Initialize semi optimized estimator
    """
    estimator = RandomForestClassifier(n_estimators=250, criterion='gini', min_samples_leaf=10, oob_score=True,
                                       n_jobs=-1)

    """
        Start RFE reduction (stop at 50 features)
    """
    train_data_reduced, ranking, rfe = reducer.reduce_RFE(train_features, train_subject_labels, estimator,
                                                          n_features_to_select=985)

    print("Saving RFE...")
    handyman.dump_pickle(rfe, results_location + "rfe.pkl")

    print("RFE produced the following ranking: ")
    print(ranking)

    print("Visualising RFE feature importance...")
    visualize_feature_importance(trainer, rfe, estimator, train_features, train_subject_labels, train_session,
                                 results_location, std=True)

if __name__ == '__main__':
    main()
