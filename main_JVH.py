import trainer
import os
from models import logistic_regression, random_forest, gradient_boosted_trees
from model_evaluation import scorer, visualiser
from feature_reduction import feature_reducer
import numpy as np
import pandas as pd
from utils import pandaman

options = ["JVH", "submission_test", "random_forest", "lda_20", "untuned"]
results_location = os.path.join("Results", '/'.join(options) + "/")
# init trainer
trainer = trainer.Trainer("")
# init model
estimator = random_forest.RF()

# load data from feature file
train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
    os.path.join("feature_extraction", '_data_sets/unreduced_transformed.pkl'), final=False)
print(train_features.shape)
print np.unique(np.array(train_subject_labels.values))
print np.unique(np.array(train_activity_labels.values))

# reduce features
reducer = feature_reducer.get_LDA_reducer(train_features, train_activity_labels, 20)
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
