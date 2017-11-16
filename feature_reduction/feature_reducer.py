from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import mutual_info_classif, f_classif


class Scorer():
    MUTUAL_INFO_CLASSIF = mutual_info_classif
    F_CLASSIF = f_classif


def reduce_feature_space(train_data, train_labels, m, p=1, score_func=None, percentile=None):
    if m == "variance":
        return __reduce_variance(train_data, p=p)
    elif m == "percentile":
        return __reduce_percentile(train_data, train_labels, score_func=score_func,
                                   percentile=percentile)
    else:
        print("No reduction, returning original training _data")
        return train_data


def __evaluate_reduction(indices_selected, n_features):
    indices_removed = set(range(0, n_features)) - set(indices_selected)
    print("The Variance reduction wants to select the following features: {}".format(indices_selected))
    print("The variance reduction wants to delete the following features: {}".format(indices_removed))


def __reduce_variance(train_data, p):
    vt = VarianceThreshold(threshold=(p * (1 - p)))
    train_data_reduced = vt.fit_transform(train_data)
    __evaluate_reduction(vt.get_support(indices=True), train_data.shape[1])
    return train_data_reduced


def __reduce_k_best(train_data, train_labels, score_func=Scorer.F_CLASSIF, k='all'):
    select_k_best = SelectKBest(score_func=score_func, k=k)
    train_data_reduced = select_k_best.fit_transform(train_data, train_labels)
    __evaluate_reduction(select_k_best.get_support(indices=True), train_data.shape[1])
    return train_data_reduced, select_k_best.scores_, select_k_best


def __reduce_percentile(train_data, train_labels, score_func=Scorer.F_CLASSIF, percentile=10):
    select_percentile = SelectPercentile(score_func=score_func, percentile=percentile)
    train_data_reduced = select_percentile.fit_transform(train_data, train_labels)
    __evaluate_reduction(select_percentile.get_support(indices=True), train_data.shape[1])
    return train_data_reduced, select_percentile.scores_, select_percentile


def __reduce_RFE(estimator, train_data, train_labels, n_features_to_select=None):
    rfe = RFE(estimator, verbose=1, n_features_to_select=n_features_to_select)
    train_data_reduced = rfe.fit_transform(train_data, train_labels)
    __evaluate_reduction(rfe.get_support(indices=True), train_data.shape[1])
    return train_data_reduced, rfe.scores_, rfe.scores_, rfe


# TODO: Translate indices with column names of panda
# TODO: Make sure column names still work correctly after a reduction is performed
# TODO: construct example
# TODO: look at visualisations for feature reductions
