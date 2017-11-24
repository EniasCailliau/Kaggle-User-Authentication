import trainer
import os
from models import logistic_regression, random_forest, gradient_boosted_trees, two_level
from model_evaluation import scorer, visualiser
from feature_reduction import feature_reducer
import numpy as np
from utils import pandaman
import pandas as pd

options = ["JVH", "two_level", "RF-RF"]
results_location = os.path.join("Results", '/'.join(options) + "/")
# init trainer
trainer = trainer.Trainer("")

# load data from feature file
train_features, train_activity_labels, train_subject_labels, test_features = trainer.load_data(
    os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)

allUsers = [1, 2, 3, 4, 5, 6, 7, 8]
allActivities = ['1', '2', '3', '4', '5', '6', '7', '12', '13', '16', '17', '24']

activityEstimator = random_forest.RF()

# assemble lists
X = []
userEstimators = []

currentFeatures = feature_reducer.reduce_LDA(train_features, train_activity_labels, 20);
X.append(currentFeatures)
trainer.get_acc_auc(activityEstimator, currentFeatures, train_activity_labels, results_location)
activityEstimator.fit(currentFeatures, train_activity_labels)

for i in range(12): #Split this up at a later time for more control
    print "Activity " + allActivities[i] +":"
    currentEstimator = random_forest.RF()
    currentData = train_features.values[train_activity_labels.values==allActivities[i]]
    currentLabels = train_subject_labels.values[train_activity_labels.values==allActivities[i]].reshape(-1)
    currentFeatures = feature_reducer.reduce_LDA(pd.DataFrame(currentData), pd.DataFrame(currentLabels), 20);
    print currentFeatures.shape
    trainer.get_acc_auc(currentEstimator, pd.DataFrame(currentFeatures), pd.DataFrame(currentLabels), os.path.join("Results", '/'.join(options) + "/" + allActivities[i] + "/"))
    currentEstimator.fit(currentFeatures, currentLabels.ravel())
    userEstimators.append(currentEstimator)
    X.append(currentFeatures)

model = two_level.TwoLevel(activityEstimator, userEstimators)
model.predict_proba(X)