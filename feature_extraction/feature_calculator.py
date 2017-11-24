from feature_extraction._data import sensor
import numpy as np
import math
import pandas as pd
from scipy import signal, stats



def __calculate_correlations(data):
    """
        Only relevant correlations are calculated
    """
    correlations = []

    # same correlations (same accelerometer)
    correlations.append(
        data[sensor.hand_accelerometer_X_axis].corr(data[sensor.hand_accelerometer_Y_axis]))
    correlations.append(
        data[sensor.hand_accelerometer_X_axis].corr(data[sensor.hand_accelerometer_Z_axis]))
    correlations.append(
        data[sensor.hand_accelerometer_Y_axis].corr(data[sensor.hand_accelerometer_Z_axis]))
    correlations.append(
        data[sensor.chest_accelerometer_X_axis].corr(data[sensor.chest_accelerometer_Y_axis]))
    correlations.append(
        data[sensor.chest_accelerometer_X_axis].corr(data[sensor.chest_accelerometer_Z_axis]))
    correlations.append(
        data[sensor.chest_accelerometer_Y_axis].corr(data[sensor.chest_accelerometer_Z_axis]))

    # inter correlations (same axis)
    correlations.append(
        data[sensor.hand_accelerometer_X_axis].corr(data[sensor.chest_accelerometer_X_axis]))
    correlations.append(
        data[sensor.hand_accelerometer_Y_axis].corr(data[sensor.chest_accelerometer_Y_axis]))
    correlations.append(
        data[sensor.hand_accelerometer_Z_axis].corr(data[sensor.chest_accelerometer_Z_axis]))
    return correlations


def __calculate_smv(data):
    """
        Only relevant smv's are calculated
    """
    smvs = []
    smvs.append(np.sqrt(data[[sensor.hand_accelerometer_X_axis, sensor.hand_accelerometer_Y_axis,
                              sensor.hand_accelerometer_Z_axis]].pow(2).sum(axis=1)).mean(axis=0))
    smvs.append(np.sqrt(data[[sensor.hand_gyroscope_X_axis, sensor.hand_gyroscope_Y_axis,
                              sensor.hand_gyroscope_Z_axis]].pow(2).sum(axis=1)).mean(axis=0))
    smvs.append(np.sqrt(data[[sensor.chest_accelerometer_X_axis, sensor.chest_accelerometer_Y_axis,
                              sensor.chest_accelerometer_Z_axis]].pow(2).sum(axis=1)).mean(axis=0))
    smvs.append(np.sqrt(data[[sensor.chest_gyroscope_X_axis, sensor.chest_gyroscope_Y_axis,
                              sensor.chest_gyroscope_Z_axis]].pow(2).sum(axis=1)).mean(axis=0))

    return smvs


def __calculate_sma(data):
    """
        Only relevant sma's are calculated
    """
    smas = []
    smas.append(data[[sensor.hand_accelerometer_X_axis, sensor.hand_accelerometer_Y_axis,
                      sensor.hand_accelerometer_Z_axis]].abs().sum(axis=1).mean(axis=0))
    smas.append(data[[sensor.hand_gyroscope_X_axis, sensor.hand_gyroscope_Y_axis,
                      sensor.hand_gyroscope_Z_axis]].pow(2).sum(axis=1).mean(axis=0))
    smas.append(data[[sensor.chest_accelerometer_X_axis, sensor.chest_accelerometer_Y_axis,
                      sensor.chest_accelerometer_Z_axis]].pow(2).sum(axis=1).mean(axis=0))
    smas.append(data[[sensor.chest_gyroscope_X_axis, sensor.chest_gyroscope_Y_axis,
                      sensor.chest_gyroscope_Z_axis]].pow(2).sum(axis=1).mean(axis=0))

    return smas


def __count_peaks(data):
    peak_lengths = []
    for x in range(12):
        peak_lengths.append(len(signal.argrelmax(data[x], order=5)[0]))

    return peak_lengths


def __calculate_basic_stats(data, multi_dimension):
    stats = []
    stats.append(np.array(data.mean()))
    stats.append(data.median())
    stats.append(data.quantile(.05))
    stats.append(data.quantile(.2))
    stats.append(data.quantile(.8))
    stats.append(data.quantile(.95))
    stats.append(data.kurtosis())
    stats.append(data.std())
    stats.append(data.skew())
    stats.append(data.mad())
    stats.append(np.sqrt(data.pow(2).mean()))
    if multi_dimension:
        stats.append(
            data.apply(lambda x: np.max(x) / math.sqrt(np.sum(x ** 2) / len(x))))
    else:
        stats.append(
            np.max(data) / math.sqrt(np.sum(data.pow(2)) / len(data)))
    return np.array(stats)


def __calculate_spectral_stats(data):
    spectral_stats = []
    # spacial_energy
    spectral_stats.extend(data.apply(lambda x: np.sum(np.power(x, 2)), axis=0))
    # spectral_entropy
    spectral_stats.extend(data.apply(lambda x: stats.entropy(x)))
    fs = 100
    frequencies = np.arange(1, len(data) + 1) * (fs / (2.0 * len(data)))
    spectral_centroid = data.apply(lambda x: np.sum(np.multiply(np.abs(x), frequencies))/np.sum(np.abs(x)))
    spectral_stats.extend(spectral_centroid)
    principal_frequency = data.apply(lambda x: frequencies[np.argmax(x)])
    spectral_stats.extend(principal_frequency)
    return spectral_stats


def __calculate_series_stats(data):
    stats = []
    stats.extend(__calculate_basic_stats(data, True).reshape(-1))
    stats.extend(__calculate_correlations(data))
    stats.extend(__calculate_smv(data))
    stats.extend(__calculate_sma(data))


    return stats

def __calculate_derivative(data, order):
    result = np.zeros((data.shape[1], data.shape[0]-order))
    for j in range(12):
        current = data.iloc[:,j]
        for i in range(order):
            current = np.ediff1d(current)
        result[j,:] = current
    return pd.DataFrame(result).transpose()

def __calculate_derivative_stats(data):
    derivative_stats = []
    for i in range(4):
        derivative_stats.extend(__calculate_basic_stats(__calculate_derivative(data, i+1), True).reshape(-1))
    return derivative_stats


def calculate_pitch_roll_stats(data):
    movements_stats = []

    pitches_hand = data.apply(lambda x: np.arctan(x[sensor.hand_accelerometer_X_axis] / np.sqrt(
        x[sensor.hand_accelerometer_Y_axis] ** 2 + x[
            sensor.hand_accelerometer_Z_axis] ** 2) * 180 / math.pi), axis=1)
    movements_stats.extend(__calculate_basic_stats(pitches_hand, False))

    rolls_hand = data.apply(lambda x: np.arctan(x[sensor.hand_accelerometer_Y_axis] /
                                                x[sensor.hand_accelerometer_Z_axis] * 180 / math.pi), axis=1)
    movements_stats.extend(__calculate_basic_stats(rolls_hand, False))

    pitches_chest = data.apply(lambda x: np.arctan(x[sensor.chest_accelerometer_X_axis] / np.sqrt(
        x[sensor.chest_accelerometer_Y_axis] ** 2 + x[
            sensor.chest_accelerometer_Z_axis] ** 2) * 180 / math.pi), axis=1)
    movements_stats.extend(__calculate_basic_stats(pitches_chest, False))

    rolls_chest = data.apply(lambda x: np.arctan(x[sensor.chest_accelerometer_Y_axis] /
                                                 x[sensor.chest_accelerometer_Z_axis] * 180 / math.pi), axis=1)
    movements_stats.extend(__calculate_basic_stats(rolls_chest, False))
    if np.isnan(movements_stats).any():
        print("problem")
        print(movements_stats)
    return movements_stats


def calculate_time_stats(data):
    time_stats = []
    time_stats.extend(__calculate_series_stats(data))
    time_stats.extend(__calculate_derivative_stats(data))
    return time_stats


def calculate_fft_stats(data):
    stats = []
    fft_coeff = np.abs(data.apply(lambda x: np.fft.fft(x), axis=0))
    # basic fft stats
    stats.extend(__calculate_series_stats(fft_coeff))
    # advanced fft stats
    stats.extend(__calculate_spectral_stats(fft_coeff))
    return stats


"""
    Features that are implicitly here:
        features[index] = np.corrcoef(realData[:,0], realData[:,2])[0,1]
        features[index] = np.corrcoef(realData[:,3], realData[:,5])[0,1]
        features[index] = np.arccos(np.average(realData[:, 0])/np.sqrt(np.sum([np.square(np.average(realData[:, x])) for x in range(3)])))
    
    
    
    Features that currently are not added (open for discussion):
    
        features[index] = np.average(np.sum(realData[:,range(0,3)],1))

"""
