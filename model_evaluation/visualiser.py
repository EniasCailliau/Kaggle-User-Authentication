import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import scorer


def plot_learning_curve(estimator, title, X, y, location, scoring, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    cv = StratifiedKFold(n_splits=4)
    train_sizes, train_scores, test_scores = sklearn.model_selection.learning_curve(
        estimator, X, y, train_sizes=train_sizes, cv=cv, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    location = location + "learning_curve.png"
    if not os.path.exists(os.path.dirname(location)):
        os.makedirs(os.path.dirname(location))

    plt.savefig(location, bbox_inches='tight', dpi=300)


def plot_ce_learning_curve(estimator, X, y, location):
    plot_learning_curve(estimator, "Learning Curve (Cross-Entropy)", X, y, scoring=scorer.logloss_evaluator,
                        location=location + "CE_")


def plot_auc_learning_curve(estimator, X, y, location):
    plot_learning_curve(estimator, "Learning Curve (AUC)", X, y, scoring=scorer.auc_evaluator,
                        location=location + "AUC_")


def plot_acc_learning_curve(estimator, X, y, location):
    plot_learning_curve(estimator, "Learning Curve (ACC)", X, y, scoring=scorer.accuracy_evaluator,
                        location=location + "ACC_")


def plot_learning_curves(estimator, X, y, location):
    plot_ce_learning_curve(estimator, X, y, location)
    plot_auc_learning_curve(estimator, X, y, location)
    plot_acc_learning_curve(estimator, X, y, location)


def plot_confusion_matrix(model, X_test, y_test, location):
    classes = ["1", "2", "3", "4", "5", "6", "7", "8"]
    y_predict = model.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_predict)

    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    location = location + "confusion_matrix.png"
    if not os.path.exists(os.path.dirname(location)):
        os.makedirs(os.path.dirname(location))

    plt.savefig(location, bbox_inches='tight', dpi=300)
