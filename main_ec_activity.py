import os

import trainer
from feature_reduction import feature_reducer
from model_evaluation import scorer
from model_evaluation import visualiser
from models.activity_prediction import gaussianNB
from utils import pandaman

options = ["ec", "activity", "gaussian", "reduction_LDA"]

results_location = os.path.join("Results", '/'.join(options) + "/")

trainer = trainer.Trainer()

estimator = gaussianNB.GaussianNB()

train_features, train_activity_labels, train_subject_labels, test_features = trainer.load_data(
    os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)

pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                     train_subject_labels=train_subject_labels, test_features=test_features)

print("Reducing using variance")
train_features_reduced = feature_reducer.reduce_variance(train_features, p=0.95)
pandaman.print_stats(train_features_reduced=train_features_reduced)

for i in range(1, 40, 2):
    print("Starting step: {}".format(i))
    print("Reducing using LDA")
    train_features_reduced = feature_reducer.reduce_LDA(train_features, train_activity_labels, n_components=i)
    pandaman.print_stats(train_features_reduced=train_features_reduced)

    print("------------------------------------------------------------------------")
    trainer.evaluate(estimator, train_features_reduced, train_activity_labels, scorer.accuracy_evaluator,
                     results_location)
    options.append(str(i))
    visualiser.plot_learning_curves(estimator, train_features, train_activity_labels, results_location)
    trainer.save_estimator(estimator, results_location)
    estimator = trainer.load_estimator(results_location)
    options.pop()
