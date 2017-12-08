from feature_reduction.feature_analyser import visualise_features_LDA
import os
from utils import handyman
import trainer as t
from feature_extraction import extractor
import pandas as pd

if __name__ == "__main__":
    #load data
    rfe = handyman.load_pickle(os.path.join("Results", "LS", "user_with_activities", "random_forest", "RFE", "augmented", "semi-optimizedrfe.pkl"))
    ranking = rfe.ranking_
    #visualise lda
    trainer = t.Trainer()
    train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = extractor.load_prepared_data_set(
        os.path.join("feature_extraction", '_data_sets', 'augmented.pkl'))
    train_data = train_features.copy()
    train_data["activities"] = train_activity_labels

    train_data = train_data.loc[:, ranking == 1]   
    features_with_act =  set(train_data.columns.values.tolist())

    rfe = handyman.load_pickle(os.path.join("Results", "LS", "user_no_activities", "random_forest", "RFE", "augmented", "semi-optimizedrfe.pkl"))
    ranking = rfe.ranking_
    train_data_without = train_features.loc[:, ranking == 1]   
    features_no_act =  set(train_data_without.columns.values.tolist())

    # print "features unique to rfe with activity"
    # print features_with_act.difference(features_no_act)
    # print "features unique to rfe without activity"
    # print features_no_act.difference(features_with_act)


    #load data
    features_FIMP =set(pd.read_csv(os.path.join("Results", "LS", "user_no_activities", "random_forest", "FIMP", "augmented", 
                "semi-optimizedforest_importance.csv")).head(150)["feature name"])
    #visualise lda
    trainer = t.Trainer()
    train_features, train_activity_labels, train_subject_labels, train_session_id, test_features = extractor.load_prepared_data_set(
        os.path.join("feature_extraction", '_data_sets', 'augmented.pkl'))

    rfe = handyman.load_pickle(os.path.join("Results", "LS", "user_no_activities", "random_forest", "RFE", "augmented", "semi-optimizedrfe.pkl"))
    ranking = rfe.ranking_
    train_data_without = train_features.loc[:, ranking == 1]   
    features_RFE =  set(train_data_without.columns.values.tolist())

    unique_RFE = features_RFE.difference(features_FIMP)
    unique_FIMP = features_FIMP.difference(features_RFE)
    print "features unique to rfe with activity"
    print unique_RFE
    print "{} features".format(len(unique_RFE))
    print "features unique to rfe without activity"
    print unique_FIMP
    print "{} features".format(len(unique_FIMP))


    # visualise_features_LDA(train_data, train_subject_labels, os.path.join("Results", "LS", "lda_rf.png"))