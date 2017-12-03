from scipy import signal
import numpy as np
import pandas as pd
from scipy.signal import filtfilt, butter
import sympy as sp
import math


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
            b, a= butter(3, 0.3)
            # Use filtfilt to apply the filter.
            data.set_value(index, "interval_data", interval_data.apply(lambda x: filtfilt(b, a, x), axis=0))

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
    elif technique == "moving_avg":
        for index in range(0, data.shape[0]):
            for x in range(12):
                interval_data = data.loc[index, "interval_data"]
                data.set_value(index, "interval_data", interval_data.apply(lambda x: moving_average_nan(x), axis=0))
    elif technique == "rolling":
        for index in range(0, data.shape[0]):
            for x in range(12):
                interval_data = data.loc[index, "interval_data"]
                data.set_value(index, "interval_data", interval_data.apply(lambda x: x.rolling(4,center=True,min_periods=1).mean(), axis=0))
            

    return data


def moving_average_nan(a, n=4):
    a = np.ma.masked_array(a,np.isnan(a))
    ret = np.cumsum(np.ma.filled(a, 0))
    ret[n:] = ret[n:] - ret[:-n]
    counts = np.cumsum(~a.mask)
    counts[n:] = counts[n:] - counts[:-n]
    ret[~a.mask] /= counts[~a.mask]
    ret[a.mask] = np.nan

    return ret

def coordinate_transform(data, technique):
    # Note that depitch might not work well when ascending or descending
    # This function is based on the assumption that the net movement is 1D and tries to align it with the positive X axis
    if technique == "depitch_all":
        for index in range(0, data.shape[0]):
            interval_data = data.loc[index, "interval_data"]
            interval_data_array = np.asarray(interval_data)

            data_sums = np.sum(interval_data_array, axis=0)

            # Accelerometer data transform
            rotate_hand = __rotate_to_x(np.array([data_sums[0], data_sums[1], data_sums[2]]))
            rotate_chest = __rotate_to_x(np.array([data_sums[6], data_sums[7], data_sums[8]]))

            interval_data_array[:, 0:3] = np.transpose(np.dot(rotate_hand, np.transpose(interval_data_array[:, 0:3])))
            interval_data_array[:, 6:9] = np.transpose(np.dot(rotate_chest, np.transpose(interval_data_array[:, 6:9])))

            data.set_value(index, "interval_data", pd.DataFrame(interval_data_array))

    elif technique == "depitch_chest":
        for index in range(0, data.shape[0]):
            interval_data = data.loc[index, "interval_data"]
            interval_data_array = np.asarray(interval_data)

            data_sums = np.sum(interval_data_array, axis=0)

            # Accelerometer data transform
            rotate_chest = __rotate_to_x(np.array([data_sums[6], data_sums[7], data_sums[8]]))

            interval_data_array[:, 6:9] = np.transpose(np.dot(rotate_chest, np.transpose(interval_data_array[:, 6:9])))

            data.set_value(index, "interval_data", pd.DataFrame(interval_data_array))
    return data


def __rotate_to_x(original_vec):
    ## STEP 1: remove z component
    current_rotation = np.identity(3)
    next_rotation = np.identity(3)
    x = original_vec[0]
    y = original_vec[1]
    z = original_vec[2]
    original_vec = np.array([x, y, z])
    vec = original_vec

    if vec[0] == 0:
        next_rotation = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    elif vec[1] == 0:
        next_rotation = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    elif vec[2] != 0:
        next_rotation = __rotation_matrix([1, 0, 0], math.atan(-vec[2] / vec[1]))

    current_rotation = np.dot(next_rotation, current_rotation)
    vec = np.dot(current_rotation, original_vec)

    if (vec[0] < 0):
        next_rotation = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    else:
        next_rotation = np.identity(3)

    current_rotation = np.dot(next_rotation, current_rotation)
    vec = np.dot(current_rotation, original_vec)

    ## STEP 2: remove y component
    next_rotation = np.identity(3)
    if vec[0] == 0:
        next_rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    elif vec[1] != 0:
        next_rotation = __rotation_matrix([0, 0, 1], math.atan(-vec[1] / vec[0]))

    current_rotation = np.dot(next_rotation, current_rotation)
    vec = np.dot(current_rotation, original_vec)

    return current_rotation


def __rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Sourced from https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector (27/11/2017)
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def getPercentiles(flat_data, percentiles, abs=False):

    results = np.zeros((12,len(percentiles)))
    all_data = [interval.values for interval in flat_data]
    all_data_array = np.concatenate(all_data, axis=0)
    if(abs):
        all_data_array = np.absolute(all_data_array)
    for j in range(len(percentiles)):
        results[:,j] = np.percentile(all_data_array, percentiles[j], axis=0)

    return results

