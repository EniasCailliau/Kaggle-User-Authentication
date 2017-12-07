import os

import numpy as np

import trainer as t
from utils import pandaman
import xgboost as xgb
import time
from utils import create_submission


ALL_USERS = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
ALL_ACTIVITIES = np.asarray([1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24])

def print_stats(test_features, train_activity_labels, train_features, train_session_id, train_subject_labels):
    pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                         train_subject_labels=train_subject_labels, train_session_id=train_session_id,
                         test_features=test_features)

def main():
    options = ["JVH", "two_level", "file"]
    results_location = os.path.join("Results", '/'.join(options) + "/")
    # init trainer
    trainer = t.Trainer("")
    train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = trainer.load_data(
        os.path.join("feature_extraction", '_data_sets/augmented.pkl'), final=False)


    # Setting up estimators
    classifiers = []
    backup_classifier = trainer.load_estimator_file(os.path.join("Estimators/XGB_one_level_994.pkl"))

    ## Activity classifier:
    classifiers.append(trainer.load_estimator_file(os.path.join("Estimators/XGB_activity_augmented_optimal_9958.pkl")))

    ## User classifier for activity 1
    classifiers.append(trainer.load_estimator_file(os.path.join("Estimators/XGB_user_1_9901.pkl")))

    ## User classifier for activity 2
    classifiers.append(trainer.load_estimator_file(os.path.join("Estimators/XGB_user_2_9758.pkl")))

    ## User classifier for activity 3
    classifiers.append(trainer.load_estimator_file(os.path.join("Estimators/XGB_user_3_9859.pkl")))

    ## User classifier for activity 4
    classifiers.append(trainer.load_estimator_file(os.path.join("Estimators/XGB_user_4_9996.pkl")))

    ## User classifier for activity 5
    classifiers.append(trainer.load_estimator_file(os.path.join("Estimators/XGB_user_5_9964.pkl")))

    ## User classifier for activity 6
    classifiers.append(trainer.load_estimator_file(os.path.join("Estimators/XGB_user_6_9915.pkl")))

    ## User classifier for activity 7
    classifiers.append(trainer.load_estimator_file(os.path.join("Estimators/XGB_user_7_9995.pkl")))

    ## User classifier for activity 12
    classifiers.append(trainer.load_estimator_file(os.path.join("Estimators/XGB_user_12_9952.pkl")))

    ## User classifier for activity 13
    classifiers.append(trainer.load_estimator_file(os.path.join("Estimators/XGB_user_13_9852.pkl")))

    ## User classifier for activity 16
    classifiers.append(trainer.load_estimator_file(os.path.join("Estimators/XGB_user_16_9621.pkl")))

    ## User classifier for activity 17
    classifiers.append(trainer.load_estimator_file(os.path.join("Estimators/XGB_user_17_9772.pkl")))

    ## User classifier for activity 24
    classifiers.append(trainer.load_estimator_file(os.path.join("Estimators/XGB_user_24_9929.pkl")))


    # Create a submission
    start = time.time()
    print "Starting prediction from file..."

    activity_probabilities = classifiers[0].predict_proba(test_features)
    activities = classifiers[0].classes_

    user_probabilities = np.zeros((len(activity_probabilities), len(ALL_USERS)))
    for i in range(len(ALL_ACTIVITIES)):
        print "Predicting " + str(ALL_ACTIVITIES[i])
        realIndex = np.where(activities == str(ALL_ACTIVITIES[i]))[0][0]

        partial_probabilities = classifiers[i + 1].predict_proba(test_features)

        current_users = classifiers[i + 1].classes_
        for j in range(len(activity_probabilities)):
            for cu in current_users:
                act_prob = activity_probabilities[j, realIndex]
                partial_prob = partial_probabilities[j, np.argwhere(current_users == cu)]
                user_probabilities[j, cu - 1] += act_prob * partial_prob

    backup_probabilities = backup_classifier.predict_proba(test_features)
    for i in range(len(activity_probabilities)):
        onelevel = backup_probabilities[i]
        twolevel = user_probabilities[i]
        if(np.argmax(onelevel) == np.argmax(twolevel) and np.amax(onelevel) > .4 and np.amax(twolevel) > .8):
            if(np.amax(onelevel) > np.amax(twolevel)):
                user_probabilities[i] = backup_probabilities[i]
        else:
            user_probabilities[i] = backup_probabilities[i]

    create_submission.write_predictions_to_csv(user_probabilities, "Predictions/two_level_from_file_12_06_19_allact_hybrid.csv")

    end = time.time()
    print(str(end - start) + "s elapsed")

if __name__ == '__main__':
    main()
