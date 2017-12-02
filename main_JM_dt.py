import os

import trainer
from model_evaluation import scorer, visualiser
from models.subject_prediction import decision_trees
from utils import pandaman
from sklearn.tree import DecisionTreeClassifier

options = ["decision_trees", "test"]
results_location = os.path.join("Results", '/'.join(options) + "/")

trainer = trainer.Trainer()

estimator = DecisionTreeClassifier(criterion='entropy', max_depth=40, min_samples_split=2, min_samples_leaf=4)

train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)

pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                     train_subject_labels=train_subject_labels, test_features=test_features)

tuned_parameters = [
    {'criterion': ['gini', 'entropy'], 'max_depth': [10, 20, 30, 40, 50], 'min_samples_split': [1, 2, 3, 4, 5],
     'min_samples_leaf': [1, 2, 3, 4, 5], }
]
#tuned_parameters = [
#    {'max_depth': [10, 20, 30, 40, 50, 60]}
#]

if __name__ == "__main__":

    #trainer.find_optimized_model(estimator, train_features, train_subject_labels, tuned_parameters, scorer.auc_evaluator)

    #trainer.evaluate(estimator, train_features, train_subject_labels, scorer.auc_evaluator, results_location)
    #trainer.evaluate(estimator, train_features, train_subject_labels, scorer.accuracy_evaluator, results_location)
    print("plotting")
    visualiser.plot_learning_curves(estimator, train_features, train_subject_labels, train_session_id, results_location)
