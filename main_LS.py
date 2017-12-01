import os

import trainer
from sklearn.svm import SVC
from model_evaluation import scorer

if __name__ == '__main__':

    options = ["LS","user_classification" ,"svm", "lda", "tuned_coarse"]
    results_location = os.path.join("Results", '/'.join(options) + "/")

    trainer = trainer.Trainer(lda=True)

    estimator = SVC(class_weight='balanced', C=100, kernel='rbf', gamma=0.0001, probability=True)
    train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets','unreduced.pkl'), final=False)

    # pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
    #                      train_subject_labels=train_subject_labels, test_features=test_features)
    #train_features = feature_reducer.reduce_k_best(train_features, train_subject_labels, k=250)[0]
    # train_features = feature_reducer.reduce_LDA(train_features, train_subject_labels, 7)

    # pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
    #                      train_subject_labels=train_subject_labels, test_features=test_features)

    # Perform parameter optimisation
    tuned_parameters = [
      {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
      {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
      {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['poly'], 'degree': [2, 3, 4]}, # coef0 is best left at 0 for lower degree polys
    #  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['sigmoid']} # paper suggesting to not use sigmoid kernel
    ]
    trainer.find_optimized_model(estimator, train_features, train_subject_labels, train_session_id, tuned_parameters, scorer.auc_evaluator)

    #print trainer.evaluate(estimator, train_features, train_subject_labels, train_session_id)
    #visualiser.plot_confusion_matrix(estimator, np.array(train_features.values), train_subject_labels, train_session_id, results_location)
    #visualiser.plot_learning_curves(estimator, train_features, train_subject_labels, train_session_id, results_location)


