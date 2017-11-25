
import numpy as np

def cv(num_folds, train_session):
        folds_indices = [[], [], [], []]
        for i in set(train_session):
            indices = train_session[train_session == i].index
            indices_splits = np.array_split(indices, num_folds)
            for j in range(num_folds):
                folds_indices[((i + j) % num_folds)].extend(indices_splits[j].values)
        custom_cv_iterator = []
        for i in range(4):
            train_indices = []
            for j in range(num_folds - 1):
                train_indices.extend(folds_indices[(i + j) % num_folds])
            test_indices = folds_indices[(i + num_folds - 1) % num_folds]

            custom_cv_iterator.append((np.array(train_indices), np.array(test_indices)))
        return custom_cv_iterator
