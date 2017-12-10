from feature_reduction import feature_analyser
import trainer as t
import os
from analysis import user_visualiser
trainer = t.Trainer()
train_features, train_activity_labels, train_subject_labels, train_sessions, test_features = trainer.load_data(
    os.path.join("feature_extraction", '_data_sets/augmented.pkl'), final=False)

feature_analyser.analyse_correlation_with_subject(train_features, train_subject_labels)


