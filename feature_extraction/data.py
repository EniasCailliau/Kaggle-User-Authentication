"""Module for loading the train and test data.

This module supports loading the data in two seperate ways.
The first, which is done by calling the load_train and load_test functions,
opens and parses the .dat files one by one, storing the content in numpy
Arrays. This way of loading the data can be slow.
The second way of loading the data is by opening a pickle file containing the
results of the load_train and load_test functions. To create this pickle file,
call the create_data_pickle function once. Afterwards, you will be able to
quickly load the data using the load_data_pickle function.
"""
import collections
import csv
import glob
import os
import warnings

import numpy as np

# Train and test _data locations
from utils import handyman

## Default _data folder names ##
# =========================== #

dir = os.path.dirname(__file__)
DEFAULT_DATA_LOCATION = os.path.join(dir, '_data')

DEFAULT_TRAIN_DATA_LOCATION = os.path.join(DEFAULT_DATA_LOCATION, 'Train')
DEFAULT_TEST_DATA_LOCATION = os.path.join(DEFAULT_DATA_LOCATION, 'Test')


# File and folder templates
_digit = '[0-9]'
SUBJECT_FOLDER_TEMPLATE = 'subject_' + _digit * 2
SESSION_FOLDER_TEMPLATE = 'session_' + _digit * 2 + '_' + _digit * 3
INTERVAL_FILE_TEMPLATE = _digit * 5 + '_' + _digit * 3 + '.dat'
TEST_FILE_TEMPLATE = _digit * 6 + '.dat'
# Metadata files
ACTIVITIES_FILE_NAME = 'activities.csv'
# _data pickle file location
DEFAULT_PICKLE_PATH = os.path.join(DEFAULT_DATA_LOCATION, 'data.pkl')

## Classes to store the parsed _data in. ##
# ====================================== #

Session = collections.namedtuple(
    'Session',
    [
        'id',  # The name of the folder containing the session _data
        'number',  # The session number for that subject
        'subject',  # The id of the subject
        'activity',  # The activity the subject is performing
        'intervals',  # Sorted list of 2s intervals of recorded _data
    ])
Interval = collections.namedtuple(
    'Interval',
    [
        'id',  # The name of the file containing the interval _data
        'time',  # The start time within the session in seconds
        'session',  # The session object
        'data',  # Numpy Array containing the raw _data
    ])


## Helper functions for loading _data ##
# =================================== #

def _load_activities(train_data_folder, fname=ACTIVITIES_FILE_NAME):
    """Loads and parses the activities csv file."""
    activities_map = {}
    with open(os.path.join(train_data_folder, fname), 'rb') as csvfile:
        freader = csv.reader(csvfile, delimiter=',')
        for ridx, row in enumerate(freader):
            # The first row is the header, and can be discarded
            if not ridx:
                continue
            ses_id, act_id = row
            activities_map[ses_id] = act_id
    return activities_map


def _get_all_session_folders(train_data_folder):
    """Retrieves all session folder names."""
    return glob.glob(os.path.join(
        train_data_folder,
        SUBJECT_FOLDER_TEMPLATE,
        SESSION_FOLDER_TEMPLATE))


def _load_session(session_folder_name, activities_map=None):
    """Loads all the _data for a particular session"""
    # Extract the session id and subject id from the folder name
    session_id = os.path.basename(session_folder_name)
    session_nr = int(session_id[-3:])
    subject_id = int(session_id[-6:-4])
    # Create the Session object
    session_activity = activities_map[session_id] if activities_map else None
    session = Session(session_id, session_nr, subject_id, session_activity, [])
    # Load each of the .dat files in the session, and sort them
    # by sorting them by name, they are automatically sorted by time.
    all_session_data_files = sorted(glob.glob(os.path.join(
        session_folder_name, INTERVAL_FILE_TEMPLATE)))
    for data_file in all_session_data_files:
        # Extract id and time from the filename
        interval_id = os.path.basename(data_file)
        interval_start_time = float(interval_id[:-len('.dat')].replace('_', '.'))
        raw_data = np.loadtxt(data_file)
        # Construct Interval object
        interval = Interval(interval_id, interval_start_time, session, raw_data)
        # Add the object to the list of intervals
        session.intervals.append(interval)
    return session


def _get_all_test_filenames(test_data_folder):
    """Retrieves all test _data file names."""
    return glob.glob(os.path.join(test_data_folder, TEST_FILE_TEMPLATE))


def _load_test_interval(test_fname):
    interval_id = os.path.basename(test_fname)
    raw_data = np.loadtxt(test_fname)
    interval = Interval(interval_id, None, None, raw_data)
    return interval


## Functions for loading and parsing all of the _data ##
# =================================================== #

def load_train(train_data_folder=DEFAULT_TRAIN_DATA_LOCATION):
    """Loads and parses the train _data.

    Args:
      train_data_folder: string containing the path to the folder containing the
          training _data.

    Returns:
      A list of all the session objects, each session containing a list of its
      _data intervals, which in turn hold the raw _data.
    """
    # First, load and parse the activities file
    activities = _load_activities(train_data_folder)

    # Second, load all the _data, for each of the sessions, for all the subjects
    all_session_folders = _get_all_session_folders(train_data_folder)
    sessions = [
        _load_session(session_folder, activities)
        for session_folder in all_session_folders]

    return sessions


def load_test(test_data_folder=DEFAULT_TEST_DATA_LOCATION):
    """Loads and parses the test _data.

    Args:
      test_data_folder: string containing the path to the folder containing the]
          test _data.

    Returns:
      A list of all the test intervals, sorted by their id.
    """
    all_fnames = sorted(_get_all_test_filenames(test_data_folder))
    all_intervals = [_load_test_interval(fname) for fname in all_fnames]
    return all_intervals


## Functions for loading the _data from a pickle file. ##
# ==================================================== #


def create_pickled_data(train_data_folder=DEFAULT_TRAIN_DATA_LOCATION,
                        test_data_folder=DEFAULT_TEST_DATA_LOCATION,
                        pickled_data_file_path=DEFAULT_PICKLE_PATH,
                        overwrite_old=True):
    """Creates the _data pickle file.

    Loads and parses the train and test _data, and then writes it to a single
    pickle file.

    Args:
      train_data_folder: path to the train _data folder.
      test_data_folder: path to the test _data folder.
      pickled_data_file_path: location where the resulting pickle file should
          be stored.
      overwrite_old: flag indicating whether the old pickle file should be
          overwritten
    """
    if os.path.exists(pickled_data_file_path):
        if not overwrite_old:
            return
        warnings.warn(
            "There already exists a data pickle file, which will be overwritten.")
    train_data = load_train(train_data_folder)
    test_data = load_test(test_data_folder)
    handyman.dump_pickle(
        dict(train=train_data, test=test_data), pickled_data_file_path)


def load_pickled_data(pickled_data_file_path=DEFAULT_PICKLE_PATH):
    """Loads the train and test _data from a pickle file.

    Args:
      pickled_data_file_path: location of the _data pickle file.
    """
    return handyman.load_pickle(pickled_data_file_path)


if __name__ == '__main__':
    """
         ATTENTION: Only execute the pickle commands in one of the main programs
    """
