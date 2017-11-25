from os import system

import trainer as t
from feature_reduction import feature_reducer as reducer
from feature_reduction.feature_reducer import Scorer
from model_evaluation import visualiser, scorer
from models.activity_prediction import xgboost_activity, random_forest
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
    # base_options = ["ec", "activity", "random_forest"]
    #
    # trainer = t.Trainer()
    # train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
    #     os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)
    # print_stats(test_features, train_activity_labels, train_features, train_session_id, train_subject_labels)
    #
    # auc_means, auc_stds, acc_means, acc_stds, numbers_of_features = [], [], [], [], []
    #
    # for k in [100, 200, 300, 400, 500, 600, 700, 800, 'all']:
    #     print("Starting with reduction to k= {}".format(k))
    #     system('say ' + 'Starting with reduction to k= {}'.format(k))
    #
    #     estimator = random_forest.RandomForest(n_estimators=200, criterion='gini', oob_score=True)
    #     options = base_options + ["best_" + str(k)] + ["n_estimators=200, criterion=gini, oob_score=True"]
    #
    #     results_location = handyman.calculate_path_from_options("Results", options)
    #     print("location: {}".format(results_location))
    #
    #     train_features_reduced, _, _ = reducer.reduce_k_best(train_features, train_activity_labels,
    #                                                          score_func=Scorer.MUTUAL_INFO_CLASSIF, k=k, verbose=1)
    #     auc_mean, auc_std, acc_mean, acc_std = evaluate(estimator, train_activity_labels, train_features_reduced,
    #                                                     train_session_id,
    #                                                     trainer)
    #     auc_means.append(auc_mean)
    #     auc_stds.append(auc_std)
    #     acc_means.append(acc_mean)
    #     acc_stds.append(acc_std)
    #     numbers_of_features.append(k)

    numbers_of_features = [100, 200, 300, 400, 500, 600, 700, 800, 994]
    auc_means = [0.98586884756318838, 0.98646719962732299, 0.98744222459578102, 0.98756145063614309,
                 0.98750270214274805, 0.98746912037334678, 0.98753359049332257, 0.98720332046812198,
                 0.98788127338263609]
    auc_stds = [0.0067255007225418386, 0.007255050484076422, 0.0058244015138245592, 0.00661082593315523,
               0.0064337114622604842, 0.0067177975952370812, 0.0063350225415548887, 0.0069947779862416437,
               0.0068258748974583732]
    acc_means = [0.90613002677695786, 0.91916944769580455, 0.92580250348648563, 0.92681282937464471, 0.92816245853762469,
                0.92973614966009188, 0.92872390254992376, 0.92973579578309917, 0.92782558247181679]
    acc_stds = [0.0066978892405828412, 0.0066458431738004361, 0.0069111570303241826, 0.0062144582692149333,
               0.0053991005299425226, 0.0057559789951326205, 0.0054192186042130155, 0.0052176248698806516,
               0.0058896304666740135]

    print("auc_means:")
    print(auc_means)
    print("auc_stds:")
    print(auc_stds)
    print("acc_means:")
    print(acc_means)
    print("acc_stds:")
    print(acc_stds)

    plt.figure()
    plt.errorbar(numbers_of_features, auc_means, yerr=auc_stds, marker='o')
    plt.title("Feature selection (SelectKBest)")
    plt.xlabel("# features")
    plt.ylabel("AuC")
    plt.savefig("Feature_importance1", bbox_inches='tight', dpi=300)

    plt.clf()
    plt.errorbar(numbers_of_features, acc_means, yerr=acc_stds, marker='o')
    plt.title("Feature selection (selectKBest")
    plt.xlabel("# features")
    plt.ylabel("Accuracy")
    plt.savefig("Feature_importance2", bbox_inches='tight', dpi=300)

    print("auc_means:")
    print(auc_means)
    print("auc_std:")
    print(auc_stds)
    print("acc_mean:")
    print(acc_means)
    print("acc_std:")
    print(acc_stds)

    system('say Your program has finished!')
    system('say Your program has finished!')
    system('say Your program has finished!')


if __name__ == '__main__':
    main()
