import trainer
import os
from models import logistic_regression, random_forest, gradient_boosted_trees
from model_evaluation import scorer, visualiser
from feature_reduction import feature_reducer
from utils import pandaman

options = ["JVH", "activity_classification", "xgboost", "lda_20", "untuned"]
results_location = os.path.join("Results", '/'.join(options) + "/")
# init trainer
trainer = trainer.Trainer("")
# init model
estimator = gradient_boosted_trees.XGB(booster='dart')

# load data from feature file
train_features, train_activity_labels, train_subject_labels, test_features = trainer.load_data(
    os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)
print(train_features.shape)
# reduce features
# train_features = feature_reducer.reduce_k_best(train_features, train_subject_labels, k=250)[0];
train_features = feature_reducer.reduce_LDA(train_features, train_activity_labels, 20)

print(train_features.shape)
trainer.get_acc_auc(estimator, train_features, train_activity_labels, results_location)
print("Plotting learning curves...")
visualiser.plot_learning_curves(estimator, train_features, train_activity_labels, results_location)
# print("Saving estimator...")
# trainer.save_estimator(estimator, results_location)

# Create a submission
# trainer.prepare_submission(estimator, test_features, options)
