import trainer
import os
from models import decision_trees
from model_evaluation import scorer, visualiser
from feature_reduction import feature_reducer
from utils import pandaman

options = ["decision_trees", "test"]
results_location = os.path.join("Results", '/'.join(options) + "/")

trainer = trainer.Trainer()

estimator = decision_trees.DecTree()

train_features, train_activity_labels, train_subject_labels, test_features = trainer.load_data(
    os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)

pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                     train_subject_labels=train_subject_labels, test_features=test_features)

tuned_parameters = [
    {'criterion': ['gini', 'entropy'], 'max_depth': [10, 20, 30, 40, 50], 'min_samples_split': [1, 2, 3, 4, 5],
     'min_samples_leaf': [1, 2, 3, 4, 5], }
]

#trainer.find_optimized_model(estimator, train_features, train_subject_labels, tuned_parameters, scorer.auc_evaluator)

trainer.evaluate(estimator, train_features, train_subject_labels, scorer.auc_evaluator, results_location)
visualiser.plot_learning_curves(estimator, train_features, train_subject_labels, results_location)