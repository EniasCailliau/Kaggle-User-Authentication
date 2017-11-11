import os

import numpy as np
import pandas as pd
from matplotlib import cm as cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold

from feature_extraction import extractor


def analyse_correlation_with_subject(train_data, train_labels):
    train_data = train_data.assign(subject_id=train_labels.values)
    pd.set_option('display.max_rows', 350)

    print(np.isnan(train_data).any(axis=0))

    correlations = train_data.corr(method='pearson')
    print (correlations['subject_id'].sort_values(ascending=False)[:6], '\t')
    print (correlations['subject_id'].sort_values(ascending=False)[-5:], '\t')


def reduce_feature_space(train_data, train_labels, m, p=1, score_func=None, percentile=None):
    if m == "variance":
        return reduce_feature_space_variance_threshold(train_data, p=p)
    elif m == "percentile":
        return reduce_feature_space_select_k_percentile(train_data, train_labels, score_func=score_func,
                                                        percentile=percentile)
    else:
        print("No reduction, returning original training _data")
        return train_data


def reduce_feature_space_variance_threshold(train_data, p):
    length_not_reduced = train_data.shape[1]
    sel = VarianceThreshold(threshold=(p * (1 - p)))
    train_data_reduced = sel.fit_transform(train_data)
    print("The Variance reduction wants to select the following features:")
    features_selected = sel.get_support(indices=True)
    print(features_selected)
    print("The variance reduction wants to delete the following features:")
    print(set(range(0, length_not_reduced)) - set(features_selected))
    return train_data_reduced


# def reduce_feature_space_select_k_best(train_data, train_labels, score_func):
#     selectKBest = SelectKBest(score_func=score_func, k='all')
#     selectKBest.fit(train_data, train_labels)
#     print("Scoring for features:")
#     print(selectKBest.scores_)
#     sorted = selectKBest.scores_
#     sorted.sort()
#     print(sorted)


def reduce_feature_space_select_k_percentile(train_data, train_labels, score_func, percentile=10):
    length_not_reduced = train_data.shape[1]
    selectPercentile = SelectPercentile(score_func=score_func, percentile=percentile)
    train_data_reduced = selectPercentile.fit_transform(train_data, train_labels)
    features_selected = selectPercentile.get_support(indices=True)
    print("The select percentile reduction wants to select the following features:")
    print(features_selected)
    print("The select percentile reduction wants to delete the following features:")
    print(set(range(0, length_not_reduced)) - set(features_selected))
    return train_data_reduced


def reduce_feature_space_RFE(estimator, train_data, train_labels, nfeatures=None):
    rfe = RFE(estimator, verbose=1, n_features_to_select=nfeatures)
    rfe.fit(train_data, train_labels)
    return rfe


def visualise_inter_feature_correlation(train_data):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(train_data.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Feature Correlation')
    fig.colorbar(cax, ticks=np.arange(-1, 1, 0.1))
    plt.show()


def visualise_features_PCA(train_data, train_labels):
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(train_data)
    for subject in range(1, 9):
        points = X_reduced[[i for i, s in enumerate(train_labels) if subject == s]]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], edgecolor='k', s=40, label='Subject ' + str(subject))

    plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
    ax.set_title("First three PCA directions")
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.savefig("test123/PCA", bbox_inches='tight', dpi=300)


def visualise_feature_distribution(X, y, index, location):
    for i in range(len(X[0])):
        plt.title('Feature ' + str(index))
        feature = X.iloc[:, index]

        labels = range(1, 9)
        plotData = []
        for i in labels:
            plotData.append(feature[y == i])

        plt.boxplot(plotData, labels=labels)
        plt.savefig(location + "/feature_distribution/" + str(index) + '.png')
        plt.clf()


def visualise_features_LDA(train_data, train_labels):
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = LinearDiscriminantAnalysis(n_components=3).fit(train_data, train_labels).transform(train_data)
    for subject in range(1, 9):
        points = X_reduced[[i for i, s in enumerate(train_labels) if subject == s]]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], edgecolor='k', s=40, label='Subject ' + str(subject))

    plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))

    ax.set_title("First three LDA directions")
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.savefig("test123/LDA", bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    """
         ATTENTION: This main block is for testing purposes only
    """
    train_data, train_labels, test_data = extractor.load_prepared_data_set(
        os.path.join("Private", 'RFE_Reduced_140.pkl'))

    analyse_correlation_with_subject(train_data, train_labels)

    # reduce_feature_space_variance_threshold(train_data, p=0.8)

    # reduce_feature_space_select_k_best(train_data, train_labels, f_classif)
    # reduce_feature_space_select_k_best(train_data, train_labels, mutual_info_classif)

    # reduce_feature_space_select_k_percentile(train_data, train_labels, f_classif)
    # reduce_feature_space_select_k_percentile(train_data, train_labels, mutual_info_classif)
    # visualise_inter_feature_correlation(pd.DataFrame(train_data))
# visualise_features_PCA(train_data, train_labels)
#     visualise_features_LDA(train_data, train_labels)
