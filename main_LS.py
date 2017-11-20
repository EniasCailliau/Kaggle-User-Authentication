import trainer
import os
from models import svm
from model_evaluation import scorer, visualiser
from feature_reduction import feature_reducer
from utils import pandaman

options = ["svm","test"]
results_location = os.path.join("Results", '/'.join(options) + "/")

trainer = trainer.Trainer()

estimator = svm.SVC(class_weight='balanced', C=100.0, kernel='rbf', gamma=0.0001)

train_features, train_activity_labels, train_subject_labels, test_features = trainer.load_data(
    os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)

pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                     train_subject_labels=train_subject_labels, test_features=test_features)



# Perform parameter optimisation
tuned_parameters = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['poly'], 'degree': [2, 3, 4]}, # coef0 is best left at 0 for lower degree polys
#  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['sigmoid']} # paper suggesting to not use sigmoid kernel
 ]
#trainer.find_optimized_model(estimator, train_features, train_subject_labels, tuned_parameters, scorer.auc_evaluator)

trainer.evaluate(estimator, train_features, train_subject_labels, scorer.auc_evaluator, results_location)

visualiser.plot_learning_curves(estimator, train_features, train_subject_labels, results_location)
