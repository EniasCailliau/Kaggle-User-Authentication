import trainer
import os
import xgboost
from model_evaluation import scorer, visualiser
from feature_reduction import feature_reducer
from utils import pandaman
import winsound


if __name__ == "__main__":
    sound = "d:/Current Projects/sounds/Finished SOUND Effect.wav"
    options = ["JM", "boosted_trees", "without_labels", "unreduced", "untuned"]
    results_location = os.path.join("Results", '/'.join(options) + "/")

    estimator = xgboost.XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=8,
        min_child_weight=1,
        gamma=0,
        subsample=0.75,
        colsample_bytree=0.75,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27
    )
    trainer = trainer.Trainer()
    train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/unreduced.pkl'), final=False)
    pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                         train_subject_labels=train_subject_labels, test_features=test_features)

    train_features = feature_reducer.reduce_PCA(train_features, 100)
    # train_features = feature_reducer.reduce_LDA(train_features, train_subject_labels, 100)

    tuned_parameters = [
        {
            #'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
            #'min_child_weight': [1,2],
            #'max_depth': [6,7,8],
            #'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
            #'subsample': [0.65, 0.7, 0.75],
            #'colsample_bytree': [0.65, 0.7, 0.75],
            'reg_lambda': [10, 50],
            #'scale_pos_weight': [0, 1, 2, 3, 4],
            #'n_estimators': [1100, 1200, 1300]
        }
    ]

    trainer.find_optimized_model(estimator, train_features, train_subject_labels, train_session_id,
                                 tuned_parameters, scorer.auc_evaluator)
    winsound.PlaySound(sound, winsound.SND_FILENAME)
    print trainer.evaluate(estimator, train_features, train_subject_labels, train_session_id)
    #visualiser.plot_learning_curves(estimator, train_features, train_subject_labels, train_session_id, results_location)
    #visualiser.plot_confusion_matrix(estimator, train_features, train_subject_labels, train_session_id, results_location)

    winsound.PlaySound(sound, winsound.SND_FILENAME)
