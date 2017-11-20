import trainer
import os
from models import nearest_neighbours
from model_evaluation import scorer, visualiser
from feature_reduction import feature_reducer
from utils import pandaman

options = ["nearest_neighbours", "test"]
results_location = os.path.join("Results", '/'.join(options) + "/")

trainer = trainer.Trainer()

estimator = nearest_neighbours.KNN(n_neighbors=5)

train_features, train_activity_labels, train_subject_labels, test_features = trainer.load_data(
    os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)

pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                     train_subject_labels=train_subject_labels, test_features=test_features)

tuned_parameters = [
    {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
]

#trainer.find_optimized_model(estimator, train_features, train_subject_labels, tuned_parameters, scorer.auc_evaluator)

trainer.evaluate(estimator, train_features, train_subject_labels, scorer.auc_evaluator, results_location)
visualiser.plot_learning_curves(estimator, train_features, train_subject_labels, results_location)