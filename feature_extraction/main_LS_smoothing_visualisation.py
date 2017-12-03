from utils import pandaman, handyman
from feature_extraction import data_loader
from feature_extraction import feature_preprocessor
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')

if __name__ == '__main__':
    train_data, test_data = data_loader.create_flat_intervals_structure()
    # print(os.path.join(".", "moving_avg_data.pkl"))
    # train_data_mvng_avg = feature_preprocessor.reduce_noise(train_data.copy(), "moving_avg")
    # handyman.dump_pickle(train_data_mvng_avg, os.path.join(".", "moving_avg_data.pkl"))
    # train_data_butter = feature_preprocessor.reduce_noise(train_data.copy(), "butter")
    # handyman.dump_pickle(train_data_butter, os.path.join(".", "butter_data.pkl"))
    # train_data_gaussian = feature_preprocessor.reduce_noise(train_data.copy(), "guassian")
    # handyman.dump_pickle(train_data_gaussian, os.path.join(".", "gaussian_data.pkl"))
    train_data_rolling = handyman.load_pickle(os.path.join(".", "rolling_data.pkl"))
    train_data_mvng_avg = handyman.load_pickle(os.path.join(".", "moving_avg_data.pkl"))
    train_data_butter = handyman.load_pickle(os.path.join(".", "butter_data.pkl"))
    train_data_gaussian = handyman.load_pickle(os.path.join(".", "gaussian_data.pkl"))
    plt.figure()
    plt.plot(np.arange(0,2,0.01), train_data.loc[20, "interval_data"].iloc[:, 0],
            label='Before smoothing',
            color='blue', linestyle=':', linewidth=2)
    plt.plot(np.arange(0,2,0.01), train_data_butter.loc[20, "interval_data"].iloc[:, 0],
        label='Butterworth',
        color='red', linewidth=1)
    plt.plot(np.arange(0,2,0.01), train_data_gaussian.loc[20, "interval_data"].iloc[:, 0],
            label='Gaussian',
            color='yellow', linewidth=1)
    plt.plot(np.arange(0,2,0.01), train_data_rolling.loc[20, "interval_data"].iloc[:, 0],
            label='Rolling average',
            color='green', linewidth=1)
    plt.title("Acceloremeter X before and after smoothing")
    plt.legend(loc="lower right")
    plt.show()