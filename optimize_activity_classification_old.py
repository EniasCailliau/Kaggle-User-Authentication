import datetime
import os
from datetime import datetime
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score

import trainer as t
from model_evaluation import visualiser, CustomKFold
from utils import pandaman
import numpy as np


def print_stats(test_features, train_activity_labels, train_features, train_session_id, train_subject_labels):
    pandaman.print_stats(train_features=train_features, train_activity_labels=train_activity_labels,
                         train_subject_labels=train_subject_labels, train_session_id=train_session_id,
                         test_features=test_features)


def plot_curves(estimator, results_location, train_labels, train_features, train_session_id):
    visualiser.plot_learning_curves(estimator, train_features, train_labels, train_session_id,
                                    results_location)
    visualiser.plot_confusion_matrix(estimator, train_features, train_labels, train_session_id, results_location)


def evaluate(estimator, train_activity_labels, train_features, train_session_id, trainer):
    auc_mean, auc_std = trainer.evaluate(estimator, train_features, train_activity_labels,
                                         train_session_id)
    acc_mean, acc_std = trainer.evaluate(estimator, train_features, train_activity_labels,
                                         train_session_id, accuracy=True)


def send_notification(subject, body):
    import smtplib
    from email.MIMEMultipart import MIMEMultipart
    from email.MIMEText import MIMEText

    fromaddr = "enias.oryx@gmail.com"
    toaddr = "enias.oryx@gmail.com"
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = subject

    body = body
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "BILLIE_2")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()


activity_to_index = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '12': 7, '13': 8, '16': 9, '17': 10,
                     '24': 11}


def main():
    try:
        options = ["EC", "activity", "XGB", "unreduced", "temp"]
        results_location = os.path.join("Results", '/'.join(options) + "/")
        # init trainer
        trainer = t.Trainer("")
        # load data from feature file
        train_features, train_activity_labels, train_subject_labels, train_sessions, test_features = trainer.load_data(
            os.path.join("feature_extraction", '_data_sets/augmented.pkl'), final=False)

        print_stats(test_features, train_activity_labels, train_features, train_sessions, train_subject_labels)
        global_start_time = datetime.now()

        train_activity_labels = train_activity_labels.apply(lambda x: activity_to_index[x])
        param = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 10,
            'min_child_weight': 1,
            'subsample': .7,
            'colsample_bytree': .7,
            'gamma': 0.05,
            'scale_pos_weight': 1,
            'nthread': 8,
            'silent': 0
        }
        # param['objective'] = 'multi:softprob'
        # param['num_class'] = 12

        estimator = xgb.XGBClassifier(**param)

        # train_index, test_index = CustomKFold.cv(4, train_sessions)[0]
        # train_X, test_X = train_features.iloc[train_index].values.astype(float), train_features.iloc[
        #     test_index].values.astype(
        #     float)
        # train_Y, test_Y = train_activity_labels.iloc[train_index].values.astype(float), train_activity_labels.iloc[
        #     test_index].values.astype(float)
        #
        # xg_train = xgb.DMatrix(train_X, label=train_Y)
        # xg_test = xgb.DMatrix(test_X, label=test_Y)
        # watchlist = [(xg_train, 'train')]
        # num_round = 5
        #
        # bst = xgb.train(param, xg_train, num_round, watchlist)
        #
        # test_dmatrix = xgb.DMatrix(test_X)
        #
        # y_scores = bst.predict(test_dmatrix, output_margin=False, ntree_limit=0)
        #
        # y_one_hot = preprocessing.label_binarize(test_Y, np.unique(test_Y))
        # print("I finally have to auc {}".format(roc_auc_score(y_one_hot, y_scores, average='macro')))
        #
        # y_pred = np.argmax(bst.predict(test_dmatrix, output_margin=False, ntree_limit=0), axis=1)
        # acc = accuracy_score(test_Y, y_pred)
        # print("I finally have to acc {}".format(acc))

        trainer = t.Trainer()
        trainer.evaluate(estimator, train_features, train_activity_labels, train_sessions)

        time_elapsed = datetime.now() - global_start_time

        print('Time elpased (hh:mm:ss.ms) {}'.format(time_elapsed))
        send_notification("Finished", 'Time elpased (hh:mm:ss.ms) {}'.format(time_elapsed))
    except Exception as e:
        print(e)
        send_notification("Exception occurred", "{}".format(e))


if __name__ == '__main__':
    main()
