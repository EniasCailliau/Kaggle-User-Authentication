from scipy import signal
import numpy as np
import pandas as pd
from scipy.signal import filtfilt, butter


def interpolate(data):
    print("STATUS: Filling NaNs")

    unfixed = 0
    for index in range(0, data.shape[0]):
        amount_before = data.loc[index, "interval_data"].isnull().sum().sum()
        if amount_before > 0:
            # First we perform a linear fill
            data.set_value(index, "interval_data",
                           data.loc[index, "interval_data"].astype(float).interpolate(
                               method='linear'))
            # Then we cover our last tracks with a backward fill (forward fill not required)
            data.set_value(index, "interval_data",
                           data.loc[index, "interval_data"].fillna(method='bfill'))
            # Code to check if any NaN remain
            amount_after = data.loc[index, "interval_data"].isnull().sum().sum()
            if amount_after > 0:
                unfixed += amount_after

    print("STATUS: Completed filling NaNs")
    print("There are {} unfixed NaN".format(unfixed))
    return data


# TODO: check this code
# TODO: Jeroen, Louis
def reduce_noise(data, technique):
    if technique == "butter":
        for index in range(0, data.shape[0]):
            interval_data = data.loc[index, "interval_data"]
            b, a, _ = butter(3, 0.3)
            # Use filtfilt to apply the filter.
            data.set_value(index, "interval_date", interval_data.apply(lambda x: filtfilt(b, a, x), axis=0))

    elif technique == "gaussian":
        for index in range(0, data.shape[0]):
            filtered_interval_data = []
            for x in range(12):
                interval_data = data.loc[index, "interval_data"]
            window = signal.general_gaussian(51, p=0.5, sig=4)
            filtered = signal.fftconvolve(window, interval_data.iloc[:, x])
            filtered = np.average(interval_data.iloc[:, x]) / np.average(filtered) * filtered
            filtered_interval_data.append(filtered)
            data.set_value(index, "interval_data", pd.DataFrame(filtered_interval_data))

    return data


