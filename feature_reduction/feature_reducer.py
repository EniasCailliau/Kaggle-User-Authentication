import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, f_classif
from utils import pandaman
import numpy as np


class Scorer():
    MUTUAL_INFO_CLASSIF = "mic"
    F_CLASSIF = "fc"


def __evaluate_reduction(indices_selected, n_features, data_frame, type):
    indices_removed = set(range(0, n_features)) - set(indices_selected)
    print("The {} reduction wants to select the following features (indices):\n {}".format(type, indices_selected))
    print("The {} reduction wants to delete the following features (indices):\n {}".format(type, indices_removed))
    print("--------------------------------------------------")
    print("The {} reduction wants to select the following features (names):\n {}".format(type,
                                                                                         pandaman.translate_column_indices(
                                                                                             indices_selected,
                                                                                             data_frame)))
    print("The {} reduction wants to delete the following features (names):\n {}".format(type,
                                                                                         pandaman.translate_column_indices(
                                                                                             np.array(
                                                                                                 list(indices_removed)),
                                                                                             data_frame)))


def visualize_RFE_ranking(ranking):
    ranking = ranking.reshape((10, -1))
    plt.matshow(ranking, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title("Ranking of features with RFE")
    plt.show()


def reduce_variance(train_data, p):
    vt = VarianceThreshold(threshold=(p * (1 - p)))
    train_data_reduced_np = vt.fit_transform(train_data)
    selected_indices = vt.get_support(indices=True)
    train_data_reduced = pd.DataFrame(train_data_reduced_np,
                                      columns=pandaman.translate_column_indices(selected_indices, train_data))
    __evaluate_reduction(vt.get_support(indices=True), train_data.shape[1], train_data, "Variance")
    return train_data_reduced


def reduce_k_best(train_data, train_labels, score_func=Scorer.F_CLASSIF, k='all'):
    if score_func == "mic":
        select_k_best = SelectKBest(score_func=mutual_info_classif, k=k)
    elif score_func == "fc":
        select_k_best = SelectKBest(score_func=f_classif, k=k)
    else:
        print("Unsupported scorer")
        return
    train_data_reduced_np = select_k_best.fit_transform(train_data, train_labels)
    selected_indices = select_k_best.get_support(indices=True)
    train_data_reduced = pd.DataFrame(train_data_reduced_np,
                                      columns=pandaman.translate_column_indices(selected_indices, train_data))
    __evaluate_reduction(selected_indices, train_data.shape[1], train_data, "selectKBest")
    return train_data_reduced, select_k_best.scores_, select_k_best


def reduce_percentile(train_data, train_labels, score_func=Scorer.F_CLASSIF, percentile=10):
    if score_func == "mic":
        select_percentile = SelectPercentile(score_func=mutual_info_classif, percentile=percentile)
    elif score_func == "fc":
        select_percentile = SelectPercentile(score_func=f_classif, percentile=percentile)
    else:
        print("Unsupported scorer")
        return
    train_data_reduced_np = select_percentile.fit_transform(train_data, train_labels)
    __evaluate_reduction(select_percentile.get_support(indices=True), train_data.shape[1], train_data,
                         "selectPercentile")
    selected_indices = select_percentile.get_support(indices=True)
    train_data_reduced = pd.DataFrame(train_data_reduced_np,
                                      columns=pandaman.translate_column_indices(selected_indices, train_data))
    return train_data_reduced, select_percentile.scores_, select_percentile


def reduce_RFE(train_data, train_labels, estimator, n_features_to_select=None):
    rfe = RFE(estimator, verbose=1, n_features_to_select=n_features_to_select)
    train_data_reduced_np = rfe.fit_transform(train_data, train_labels)
    selected_indices = rfe.get_support(indices=True)
    train_data_reduced = pd.DataFrame(train_data_reduced_np,
                                      columns=pandaman.translate_column_indices(selected_indices, train_data))
    __evaluate_reduction(rfe.get_support(indices=True), train_data.shape[1], train_data, "RFE")
    return train_data_reduced, rfe.scores_, rfe.scores_, rfe

# TODO: Make sure column names still work correctly after a reduction is performed
# TODO: construct example
# TODO: look at visualisations for feature reductions
