import trainer
import os
from models import logistic_regression, random_forest
from model_evaluation import scorer, visualiser

options = ["logistic_regression", "dd", "NO_SMOTE", "Optimized"]
results_location = os.path.join("Results", '/'.join(options))
# init trainer
trainer = trainer.Trainer("")
# init model
estimator = logistic_regression.LogReg()

# load data from feature file
train_features, train_activity_labels, train_subject_labels, test_features = trainer.load_data(
    os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)

# Perform parameter optimisation
# tuned_parameters = {'class_weight': ['balanced', None],
#                     'C': [10, 100, 1000],
#                     'multi_class': ['ovr', 'multinomial'],
#                     'solver': ['lbfgs', 'newton-cg', 'saga']}
# trainer.find_optimized_model(estimator, train_features, train_subject_labels, tuned_parameters, scorer.auc_evaluator)

# results_location is used to point to a folder which will be used to output
# trainer.evaluate(estimator, train_features, train_subject_labels, scorer.auc_evaluator, results_location)

visualiser.plot_learning_curves(estimator, train_features, train_subject_labels, results_location)

# Create a submission
# trainer.prepare_submission(estimator, test_features, options)
