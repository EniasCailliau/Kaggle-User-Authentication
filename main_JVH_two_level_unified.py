import os

import numpy as np
import pandas as pd

import trainer
from models import two_level_unified

options = ["JVH", "two_level", "RF-RF2"]
results_location = os.path.join("Results", '/'.join(options) + "/")

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
model = two_level_unified.TwoLevel()
print "TESTING"


#model.fit(X.values, y.values)
#trainer.prepare_submission(model, test.values, options)

