import pandas as pd
import numpy as np

import data

def create_flat_intervals_structure(augmented=False):
    if (augmented):
        return create_augmented_flat_intervals_structure()

    print("STATUS: Creating flat interval structure (TRAINING)")

    dataset = data.load_pickled_data()

    training_data = []

    session_id = 0
    for session in dataset['train']:
        print("session: {}".format(session_id))
        interval_id = 0
        for interval in session.intervals:
            training_data_entry = [session_id, interval.session.subject,
                                   interval.session.activity,
                                   pd.DataFrame(interval.data)]
            interval_id += 1
            training_data.append(training_data_entry)
        session_id += 1

    train_flat = pd.DataFrame(training_data, columns=["session_id", "subject", "activity", "interval_data"])

    print("STATUS: Creating flat interval structure (TEST)")

    test_data = []

    for interval in dataset['test']:
        test_data_entry = [pd.DataFrame(interval.data)]
        test_data.append(test_data_entry)

    test_flat = pd.DataFrame(test_data, columns=["interval_data"])

    return train_flat, test_flat

def create_augmented_flat_intervals_structure():
    print "Augmenting data..."
    print("STATUS: Creating flat interval structure (TRAINING)")

    dataset = data.load_pickled_data()

    training_data = []

    session_id = 0
    for session in dataset['train']:
        print("session: {}".format(session_id))
        raw_session = np.concatenate([interval.data for interval in session.intervals[::2]],axis=0)
        for i in range(0,len(raw_session)-200,20):
            current_data = raw_session[i:i+200]
            training_data_entry = [session_id, session.subject,
                                   session.activity,
                                   pd.DataFrame(current_data)]
            training_data.append(training_data_entry)
        session_id += 1

    train_flat = pd.DataFrame(training_data, columns=["session_id", "subject", "activity", "interval_data"])

    print("STATUS: Creating flat interval structure (TEST)")

    test_data = []

    for interval in dataset['test']:
        test_data_entry = [pd.DataFrame(interval.data)]
        test_data.append(test_data_entry)

    test_flat = pd.DataFrame(test_data, columns=["interval_data"])

    return train_flat, test_flat

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

if __name__ == '__main__':
    """
        ATTENTION: This main block is for testing purposes only
    """
