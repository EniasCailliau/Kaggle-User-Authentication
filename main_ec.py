import trainer
import os
from model_evaluation import scorer, visualiser
from feature_reduction import feature_reducer
from models.activity_prediction import random_forest
from utils import pandaman, handyman

options = ["ec", "user", "randomforest", "n_ensembles=250", "cv_V0.0"]

trainer = trainer.Trainer()

estimator = random_forest.RandomForest(n_estimators=250)

train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
    os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)

pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                     train_subject_labels=train_subject_labels, train_session_id=train_session_id,
                     test_features=test_features)


#
results_location = handyman.calculate_path_from_options("Results", options)
# print("location: {}".format(results_location))
#
auc, acc = trainer.evaluate(estimator, train_features, train_subject_labels, train_session_id, results_location)
print("I have auc: {} ".format(auc))
print("I have acc: {} ".format(acc))

print("----------")
train_features = train_features.join(train_activity_labels)
#
results_location = handyman.calculate_path_from_options("Results", options)
# print("location: {}".format(results_location))
#
auc, acc = trainer.evaluate(estimator, train_features, train_subject_labels, train_session_id, results_location)
print("I have auc: {} ".format(auc))
print("I have acc: {} ".format(acc))
# visualiser.plot_learning_curves(estimator, train_features, train_subject_labels, train_session_id, results_location)
