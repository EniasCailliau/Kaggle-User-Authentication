import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_extraction import data_loader

activity_to_index = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '12': 7, '13': 8, '16': 9, '17': 10,
                     '24': 11}
column_titles = np.sort(np.array(activity_to_index.keys(), dtype=int))


def analyse_nan_distributions(training_intervals_flat):
    total_number_of_nans = 0
    total_nans_per_category = np.zeros(12)
    total_nans_per_subject = np.zeros(8)
    total_nans_per_subject_per_activity = np.zeros((8, 12))
    streaks = []
    for index, interval in training_intervals_flat.iterrows():
        data = interval["_data"]
        nans_per_category = data.isnull().sum(axis=0).values
        max_nan_streak_category = []
        for i in range(0, 12):
            m = np.max([sum(g) for b, g in itertools.groupby(data.isnull()[2]) if b] + [0])
            max_nan_streak_category.append(m)
        streaks.append(max_nan_streak_category)
        total_nans_per_category += nans_per_category
        number_of_nans = nans_per_category.sum()
        total_number_of_nans += number_of_nans
        total_nans_per_subject[interval["subject"] - 1] += number_of_nans
        total_nans_per_subject_per_activity[
            interval["subject"] - 1, activity_to_index[interval["activity"]]] += total_number_of_nans

    print("max streaks")
    print(pd.DataFrame(streaks))
    print(total_number_of_nans)
    print(total_nans_per_category)

    df_nans_per_subject = pd.DataFrame(total_nans_per_subject, columns=['# NaN'])
    df_nans_per_subject.index = [1, 2, 3, 4, 5, 6, 7, 8]
    df_nans_per_subject.plot.bar()
    plt.draw()

    df_nans_per_subject_per_category = pd.DataFrame(total_nans_per_subject_per_activity, columns=column_titles)
    df_nans_per_subject_per_category.plot.bar(stacked=True)
    plt.draw()

    plt.figure()
    streaks_dataframe = pd.DataFrame(streaks)

    streaks_dataframe[(streaks_dataframe.T != 0).any()].boxplot()
    plt.draw()


def analyse_interval_distributions(training_intervals_flat):
    total_intervals_per_subject = np.zeros(8)
    total_intervals_per_subject_per_category = np.zeros((8, 12))
    for index, interval in training_intervals_flat.iterrows():
        total_intervals_per_subject[interval["subject"] - 1] += 1
        total_intervals_per_subject_per_category[interval["subject"] - 1, activity_to_index[interval["activity"]]] += 1

    df_intervals_per_subject = pd.DataFrame(total_intervals_per_subject, columns=['# intervals'])
    # df_intervals_per_subject.plot.bar()
    # plt.draw()

    df_intervals_per_subject_per_category = pd.DataFrame(total_intervals_per_subject_per_category,
                                                         columns=column_titles)
    print(df_intervals_per_subject_per_category.shape)
    print(df_intervals_per_subject_per_category)
    df_intervals_per_subject_per_category.index = [1, 2, 3, 4, 5, 6, 7, 8]
    print(df_intervals_per_subject_per_category)
    df_intervals_per_subject_per_category.plot.bar(stacked=True)
    plt.draw()

    # df_intervals_per_subject_per_category.plot.bar(stacked=False)
    # plt.draw()

    print(df_intervals_per_subject_per_category.isin([0, 0]))
    for i in range(0, 8):
        for j in range(0, 12):
            if not total_intervals_per_subject_per_category[i, j]:
                print("Subject {} has no intervals for category {}".format(i, column_titles[j]))

    # plt.figure()
    # df_intervals_per_subject["# intervals"].plot.pie(labels=['1', '2', '3', '4', '5', '6', '7', '8'],
    #                 autopct='%.1f', fontsize=20, figsize=(4, 4))
    # plt.tight_layout()
    # plt.draw()

    fig, axes = plt.subplots(2, 4)
    for i, ax in enumerate(axes.flatten()):
        df_intervals_per_subject_per_category.iloc[i, :].plot(kind='pie', autopct='%.1f', labels=column_titles, ax=ax,
                                                              colormap='Paired', title="Subject {}".format(i + 1),
                                                              fontsize=10)
    plt.tight_layout()
    plt.draw()


if __name__ == '__main__':
    intervals_training = data_loader.create_flat_intervals_structure_training()
    analyse_nan_distributions(intervals_training)
    # analyse_interval_distributions(intervals_training)
    plt.show()
