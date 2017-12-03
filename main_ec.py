import os

import trainer as t
from model_evaluation import visualiser
from utils import pandaman
from model_evaluation import visualiser, scorer
import xgboost
from datetime import datetime
from utils import handyman
import numpy as np

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
    base_options = ["ec", "activity", "xgboost"]
    startTime = datetime.now()
    trainer = t.Trainer()
    # train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
    #     os.path.join("feature_extraction", '_data_sets/augmented.pkl'), final=False)

    ranker = handyman.load_pickle("/Users/eniascailliau/Documents/GitHub.nosync/Kaggle-User-Authentication/Results/LS/user_with_activities/random_forest/RFE/augmented/semi-optimizedrfe.pkl")
    np.set_printoptions(threshold='nan')

    print(ranker.ranking_)




    # print(train_session_id)
    # print_stats(test_features, train_activity_labels, train_features, train_session_id, train_subject_labels)
    # #
    # # auc_means, auc_stds, acc_means, acc_stds, numbers_of_features = [], [], [], [], []
    # #
    # #
    # estimator = xgboost.XGBClassifier(n_estimators=400, max_depth=10)
    # # options = base_options + ["unreduced"] + ["n_estimators=200, max_depth=8"]
    # #
    # # results_location = handyman.calculate_path_from_options("Results", options)
    # # print("location: {}".format(results_location))
    # #
    # # # print("K best")
    # # # train_features_reduced, _, _ = reducer.reduce_k_best(train_features, train_activity_labels,
    # # #                                                      Scorer.MUTUAL_INFO_CLASSIF, k=600)
    # #
    # param_grid = [{'n_estimators': [150, 200, 250, 300, 400]
    #                }]
    # trainer.find_optimized_model(estimator, train_features, train_activity_labels, train_session_id, param_grid,
    #                              scorer.auc_evaluator)
    # auc_mean, auc_std, acc_mean, acc_std = evaluate(estimator, train_activity_labels, train_features,
    #                                                 train_session_id,
    #                                                 trainer)
    # print("Auc: {}".format(auc_mean))
    # print("Acc: {}".format(acc_mean))
    #
    # timeElapsed = datetime.now() - startTime
    #
    # print('Time elpased (hh:mm:ss.ms) {}'.format(timeElapsed))
    os.system('say Your program has finished!')
    os.system('say Your program has finished!')
    os.system('say Your program has finished!')


if __name__ == '__main__':
    main()
