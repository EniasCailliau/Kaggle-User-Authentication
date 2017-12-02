import os

import trainer
from model_evaluation import scorer
from model_evaluation import visualiser
from models.subject_prediction import nearest_neighbours
from sklearn.neighbors import KNeighborsClassifier
from utils import pandaman

if __name__ == "__main__":
    options = ["nearest_neighbours", "test"]
    results_location = os.path.join("Results", '/'.join(options) + "/")

    trainer = trainer.Trainer()

    k=9
    string = "Results for value k = "+str(k)
    print(string)
    estimator = KNeighborsClassifier(n_neighbors=k)
    train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)
    pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                         train_subject_labels=train_subject_labels, test_features=test_features)
    #trainer.evaluate(estimator, train_features, train_subject_labels, scorer.auc_evaluator, results_location)
    #trainer.evaluate(estimator, train_features, train_subject_labels, scorer.accuracy_evaluator, results_location)

    visualiser.plot_learning_curves(estimator, train_features, train_subject_labels, train_session_id, results_location)
