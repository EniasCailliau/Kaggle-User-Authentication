import pandas as pd

from feature_extraction import data


def create_flat_intervals_structure():
    print("STATUS: Creating flat interval structure (TRAINING)")

    dataset = data.load_pickled_data()

    training_data = []

    for session in dataset['train']:
        for interval in session.intervals:
            training_data_entry = [interval.session.subject, interval.session.activity, pd.DataFrame(interval.data)]
            training_data.append(training_data_entry)

    train_flat = pd.DataFrame(training_data, columns=["subject", "activity", "interval_data"])

    print("STATUS: Creating flat interval structure (TEST)")

    test_data = []

    for interval in dataset['test']:
        test_data_entry = [pd.DataFrame(interval.data)]
        test_data.append(test_data_entry)

    test_flat = pd.DataFrame(test_data, columns=["interval_data"])

    return train_flat, test_flat


if __name__ == '__main__':
    """
        ATTENTION: This main block is for testing purposes only
    """
    train_flat, test_flat = create_flat_intervals_structure()
    print(train_flat)
    print(test_flat)
