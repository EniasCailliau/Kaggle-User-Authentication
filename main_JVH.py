import os

import numpy as np

import trainer as t
from model_evaluation import visualiser
from models.subject_prediction import random_forest
from sklearn import ensemble
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
    options = ["JVH", "random_forest", "augmented", "512_estimators"]
    results_location = os.path.join("Results", '/'.join(options) + "/")
    # init trainer
    trainer = t.Trainer("")
    # init model
    estimator = ensemble.RandomForestClassifier(n_estimators=512, n_jobs=-1, oob_score=True)

    # load data from feature file
    train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/augmented.pkl'), final=False)
    print(train_features.shape)

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
    trainer.evaluate(estimator, train_features, train_subject_labels, train_session_id)
    estimator.fit(train_features, train_subject_labels)
    trainer.prepare_submission(estimator, test_features, options)


if __name__ == '__main__':
    main()
