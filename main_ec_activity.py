import os

import trainer
from feature_reduction import feature_reducer
from model_evaluation import scorer
from model_evaluation import visualiser
from models.activity_prediction import gaussianNB, bernoulliNB, svc
from utils import pandaman, handyman

options = ["ec", "activity", "Bernoulli", "not-optimized", "reduction_LDA"]

trainer = trainer.Trainer()

# estimator = gaussianNB.GaussianNB()
estimator = bernoulliNB.Bernoulli()
# estimator = svc.SVC()

train_features, train_activity_labels, train_subject_labels, test_features = trainer.load_data(
    os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)

pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                     train_subject_labels=train_subject_labels, test_features=test_features)

print("Reducing using variance")
train_features_reduced = feature_reducer.reduce_variance(train_features, p=0.95)
pandaman.print_stats(train_features_reduced=train_features_reduced)

for i in range(5, 40, 2):
    print("Starting step: {}".format(i))
    print("Reducing using LDA")
    train_features_reduced = feature_reducer.reduce_LDA(train_features, train_activity_labels, n_components=i)
    pandaman.print_stats(train_features_reduced=train_features_reduced)

    print("------------------------------------------------------------------------")
    options.append(str(i))
    results_location = handyman.calculate_path_from_options("Results", options)
    options.pop()

    handyman.calculate_path_from_options("Results", options)


    trainer.evaluate(estimator, train_features_reduced, train_activity_labels,
                     results_location)

    visualiser.plot_learning_curves(estimator, train_features, train_activity_labels, results_location)
    trainer.save_estimator(estimator, results_location)
    estimator = trainer.load_estimator(results_location)
