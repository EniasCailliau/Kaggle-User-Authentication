import os

import numpy as np

import trainer as t
from model_evaluation import visualiser
from models.subject_prediction import random_forest
from utils import pandaman


def print_stats(test_features, train_activity_labels, train_features, train_session_id, train_subject_labels):
    pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                         train_subject_labels=train_subject_labels, train_session_id=train_session_id,
                         test_features=test_features)


def plot_curves(estimator, results_location, train_labels, train_features, train_session_id):
    visualiser.plot_learning_curves(estimator, train_features, train_labels, train_session_id,
                                    results_location)
    visualiser.plot_confusion_matrix(estimator, train_features, train_labels, train_session_id, results_location)


def evaluate(estimator, train_activity_labels, train_features, train_session_id, trainer):
    auc_mean, auc_std = trainer.evaluate(estimator, train_features, train_activity_labels,
                                                            train_session_id)
    acc_mean, acc_std = trainer.evaluate(estimator, train_features, train_activity_labels,
                                                            train_session_id, accuracy=True)

def main():
    options = ["JVH", "submission_test", "random_forest", "unreduced_with_bins", "untuned"]
    results_location = os.path.join("Results", '/'.join(options) + "/")
    # init trainer
    trainer = t.Trainer("")
    # init model
    estimator = random_forest.RF()

    # load data from feature file
    train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/unreduced_with_bins.pkl'), final=False)
    print(train_features.shape)
    print np.unique(np.array(train_subject_labels.values))
    print np.unique(np.array(train_session_id.values))

    # reduce features
    # train_features = pd.DataFrame(reducer.transform(train_features))
    # test_features = pd.DataFrame(reducer.transform(test_features))

    # print(train_features.shape)
    # trainer.get_acc_auc(estimator, train_features, train_activity_labels, results_location)
    # print("Plotting learning curves...")
    # visualiser.plot_learning_curves(estimator, train_features, train_activity_labels, results_location)
    # print("Saving estimator...")
    # trainer.save_estimator(estimator, results_location)

    # Create a submission
    # auc_mean, auc_std, acc_mean, acc_std = evaluate(estimator, train_subject_labels, train_features, train_session_id, trainer)
    estimator.fit(train_features, train_subject_labels)
    trainer.prepare_submission(estimator, test_features, options)


if __name__ == '__main__':
    main()
