import trainer
import os
from models import logistic_regression
from model_evaluation import scorer, visualiser
from feature_reduction import feature_reducer
from utils import pandaman

options = ["logistic_regression", "test"]
results_location = os.path.join("Results", '/'.join(options) + "/")

trainer = trainer.Trainer()

estimator = logistic_regression.LogReg()

train_features, train_activity_labels, train_subject_labels, test_features = trainer.load_data(
    os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)

pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                     train_subject_labels=train_subject_labels, test_features=test_features)

train_features_reduced, scores_, select_k_best = feature_reducer.reduce_k_best(train_features, train_subject_labels,
                                                                               feature_reducer.Scorer.F_CLASSIF, k=10)

pandaman.print_stats(train_features_reduced=train_features_reduced)

# trainer.find_optimized_model(estimator, train_features, train_subject_labels, tuned_parameters, scorer.auc_evaluator)

# trainer.evaluate(estimator, train_features, train_subject_labels, scorer.auc_evaluator, results_location)

# visualiser.plot_learning_curves(estimator, train_features, train_subject_labels, results_location)


# trainer.save_estimator(estimator, results_location)

# estimator = trainer.load_estimator(results_location, verbose=1)
