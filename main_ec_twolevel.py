import os

import numpy as np
from sklearn.model_selection import train_test_split

import trainer as t
from model_evaluation import visualiser
from models import two_staged_classifier
from utils import pandaman, handyman


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


"""

"""

options = ["ec", "user", "twostaged"]

results_location = handyman.calculate_path_from_options("Results", options)

trainer = t.Trainer()
train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
    os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)

print_stats(test_features, train_activity_labels, train_features, train_session_id, train_subject_labels)

train_features = train_features.join(train_activity_labels)

estimator = two_staged_classifier.twinzy()

# X_train, X_test, y_train, y_test = train_test_split(train_features, train_subject_labels, test_size=0.25)
# estimator.fit(np.array(X_train.values), np.array(y_train.values))
#
# trainer.save_estimator(estimator, results_location)
# estimator = trainer.load_estimator(results_location)


trainer.evaluate(estimator, train_features, train_subject_labels, train_session_id)



#
# print("is there a nan")
# print(X_test.isnull().values.any())
# X_real = X_test.loc[:, X_test.columns != 'activity']
# column = np.full((X_real.shape[0]),1)
# X_with_activity = X_real.assign(activity=np.full((X_real.shape[0]),1))
# print(X_with_activity.shape)
# print("is there a nan")
# print(X_with_activity.isnull().any())
#
# print(X_with_activity.iloc[:,-1])
