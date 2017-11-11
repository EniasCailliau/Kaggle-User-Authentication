import pandas as pd

import feature_calculator as calculator
import feature_preprocessor as preprocessor
from feature_extraction import data_loader
from utils import handyman as utils


def generate_column_names():
    feature_names = []
    for metric in ["mean", "median", "q05", "q20", "q80", "q95", "kurtosis", "std", "skew", "mad", "rms",
                   "crestfactor"]:
        for placement in ["hand", "chest"]:
            for type in ["aX", "aY", "aZ", "gX", "gY", "gZ"]:
                feature_names.append(metric + "_" + type + "_" + placement + "(t)")

    feature_names.extend((s + "(t)" for s in (
        "corr_haX_haY", "corr_haX_haZ", "corr_haY_haZ", "corr_caX_caY", "corr_caX_caZ", "corr_caY_caZ",
        "corr_haX_caX", "corr_haY_caY", "corr_haZ_caZ")))
    feature_names.extend((s + "(t)" for s in ("smv_ha", "smv_hg", "smv_ca", "smv_cg")))
    feature_names.extend((s + "(t)" for s in ("sma_ha", "sma_hg", "sma_ca", "sma_cg")))

    for metric in ["mean", "median", "q05", "q20", "q80", "q95", "kurtosis", "std", "skew", "mad", "rms",
                   "crestfactor"]:
        for bodypart in ["hand", "chest"]:
            for type in ["pitch", "roll"]:
                feature_names.append(bodypart + "_" + type + "_" + metric)

    for metric in ["mean", "median", "q05", "q20", "q80", "q95", "kurtosis", "std", "skew", "mad", "rms",
                   "crestfactor"]:
        for placement in ["hand", "chest"]:
            for type in ["aX", "aY", "aZ", "gX", "gY", "gZ"]:
                feature_names.append(metric + "_" + type + "_" + placement + "(s)")

    feature_names.extend((s + "(s)" for s in (
        "corr_haX_haY", "corr_haX_haZ", "corr_haY_haZ", "corr_caX_caY", "corr_caX_caZ", "corr_caY_caZ",
        "corr_haX_caX", "corr_haY_caY", "corr_haZ_caZ")))
    feature_names.extend((s + "(s)" for s in ("smv_ha", "smv_hg", "smv_ca", "smv_cg")))
    feature_names.extend((s + "(s)" for s in ("sma_ha", "sma_hg", "sma_ca", "sma_cg")))

    for metric in ["spectral_energy", "spectral_entropy"]:
        for placement in ["hand", "chest"]:
            for type in ["aX", "aY", "aZ", "gX", "gY", "gZ"]:
                feature_names.append(metric + "_" + type + "_" + placement)

    return feature_names


def __create_data_set(noise_reducer_method="None"):
    train_flat, test_flat = data_loader.create_flat_intervals_structure()

    # First preprocessing data
    train_flat = preprocessor.interpolate(train_flat)
    test_flat = preprocessor.interpolate(test_flat)

    # TODO: this can have some other preprocessing
    preprocessor.reduce_noise(train_flat, noise_reducer_method)

    train_features = generate_features(train_flat["interval_data"])
    train_activity_labels = train_flat["activity"]
    train_subject_labels = train_flat["subject"]

    test_features = generate_features(test_flat["interval_data"])

    return [train_features, train_activity_labels, train_subject_labels, test_features]


def generate_features(data):
    print("STATUS: Deriving features from intervals")
    intervals = []
    feature_names = generate_column_names()

    for index, interval in data.iteritems():
        print(index)
        if index > 10:
            break

        interval_entry_new_features = []

        interval_entry_new_features.extend(calculator.calculate_time_stats(interval))

        interval_entry_new_features.extend(calculator.calculate_pitch_roll_stats(interval))

        interval_entry_new_features.extend(calculator.calculate_fft_stats(interval))

        if pd.isnull(interval_entry_new_features).any():
            print("!!! problem: there are NaN in the feature space !!!")

        intervals.append(interval_entry_new_features)

    return pd.DataFrame(intervals, columns=feature_names)


def prepare_data_pickle(file_path, noise_reducer_method="None"):
    train_features, train_activity_labels, train_subject_labels, test_features = __create_data_set(noise_reducer_method)
    utils.dump_pickle(
        dict(train_features=train_features, train_activity_labels=train_activity_labels,
             train_subject_labels=train_subject_labels, test_features=test_features), file_path)


def load_prepared_data_set(file_path):
    data = utils.load_pickle(file_path)
    return data["train_features"], data["train_activity_labels"], data["train_subject_labels"], data["test_features"]


if __name__ == '__main__':
    """
         ATTENTION: This main block is for testing purposes only
    """
    prepare_data_pickle("_data_sets/unreduced.pkl")
    data = utils.load_pickle("_data_sets/unreduced.pkl")
    print(data["train_features"].shape)
    print(data["train_activity_labels"].shape)
    print(data["train_subject_labels"].shape)
    print(data["test_features"].shape)
    print("ok?")
