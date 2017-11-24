import trainer
import os
from models import logistic_regression, random_forest, gradient_boosted_trees, two_level_basic
from model_evaluation import scorer, visualiser
from feature_reduction import feature_reducer
import numpy as np
from utils import pandaman
import pandas as pd

# init trainer
trainer = trainer.Trainer("")

# Load data from feature file
train_features, train_activity_labels, train_subject_labels, test_features = trainer.load_data(
    os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)

# Merge features with activity labels (required to use trainer)
X = pd.DataFrame(np.hstack([train_features.values, train_activity_labels.values.reshape(-1,1).astype(int)]))
y = train_subject_labels
test = pd.DataFrame(np.hstack([test_features.values, np.zeros((test_features.shape[0],1))]))

# Set up model
model = two_level_basic.TwoLevel()
print "TESTING"
model.fit(X.values,y.values)
model.predict_proba(test.values)

