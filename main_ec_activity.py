import os

import trainer as t
from model_evaluation import scorer
from models.activity_prediction import random_forest
from utils import pandaman, handyman


def main():
    options = ["ec", "user", "randomforest", "n_ensembles=10", "cv_V0.0"]

    trainer = t.Trainer()

    # estimator = gaussianNB.Gaussian()
    # estimator = bernoulliNB.Bernoulli()
    # estimator = svc.SVC()
    estimator = random_forest.RandomForest(verbose=0, n_estimators=10)
    # estimator = gradient_boosted_trees.XGB(max_depth=0)

    train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)


    pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                         train_subject_labels=train_subject_labels, train_session_id=train_session_id, test_features=test_features)

    # print("Reducing using variance")
    # train_features_reduced = feature_reducer.reduce_variance(train_features, p=0.95)
    # pandaman.print_stats(train_features_reduced=train_features_reduced)

    results_location = handyman.calculate_path_from_options("Results", options)
    print("location: {}".format(results_location))
    # train_features_reduced = feature_reducer.reduce_LDA(train_features, train_subject_labels, n_components=10)

    print("start visualisation")
    # visualiser.plot_learning_curves(estimator, train_features, train_subject_labels, results_location)
    # auc, acc = trainer.evaluate(estimator, train_features, train_subject_labels,
    #                             results_location)
    # print("I have {} and {}".format(auc, acc))






'''
best_acc = 0
best_lda = -1
for i in range(1, 12):
    print("Starting step: {}".format(i))
    print("Reducing using LDA")
    train_features_reduced = feature_reducer.reduce_LDA(train_features, train_activity_labels, n_components=i)
    pandaman.print_stats(train_features_reduced=train_features_reduced)

    print("------------------------------------------------------------------------")
    options.append(str(i))
    results_location = handyman.calculate_path_from_options("Results", options)
    options.pop()

    handyman.calculate_path_from_options("Results", options)

    auc, acc = trainer.evaluate(estimator, train_features_reduced, train_activity_labels,
                                results_location)
    if (best_acc < acc):
        print("updating to acc {} for lda {}".format(acc, i))
        best_acc = acc
        best_lda = i

    visualiser.plot_learning_curves(estimator, train_features_reduced, train_activity_labels, results_location)
    trainer.save_estimator(estimator, results_location)
    estimator = trainer.load_estimator(results_location)

print("The best LDA was: {}, with accuracy: {}".format(best_lda, best_acc))
'''

if __name__ == '__main__':
    main()
