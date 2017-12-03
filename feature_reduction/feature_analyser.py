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
from feature_reduction import feature_reducer


def analyse_correlation_with_subject(train_data, train_labels):
    train_data = train_data.assign(subject_id=train_labels.values)
    pd.set_option('display.max_rows', 350)

    print(np.isnan(train_data).any(axis=0))

    correlations = train_data.corr(method='pearson')
    print (correlations['subject_id'].sort_values(ascending=False)[:6], '\t')
    print (correlations['subject_id'].sort_values(ascending=False)[-5:], '\t')


def visualise_inter_feature_correlation(train_data):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(train_data.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Feature Correlation')
    fig.colorbar(cax, ticks=np.arange(-1, 1, 0.1))
    plt.show()


def visualise_features_PCA(train_data, train_labels, location):
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = feature_reducer.reduce_PCA(train_data, n_components=3, nparray=True)
    for subject in range(1, 9):
        points = X_reduced[[i for i, s in enumerate(train_labels) if subject == s]]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], edgecolor='k', s=40, label='Subject ' + str(subject))

    plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
    ax.set_title("First three PCA directions")
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.savefig(location, bbox_inches='tight', dpi=300)
    return X_reduced


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


def visualise_features_LDA(train_data, train_labels, location):
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = feature_reducer.reduce_LDA(train_data, train_labels, n_components=3, np_array=True)
    for subject in range(1, 9):
        points = X_reduced[[i for i, s in enumerate(train_labels) if subject == s]]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], edgecolor='k', s=40, label='Subject ' + str(subject))

    plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))

    ax.set_title("First three LDA directions")
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.savefig(location, bbox_inches='tight', dpi=300)
    return X_reduced


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
