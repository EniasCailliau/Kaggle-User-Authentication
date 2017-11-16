import numpy as np
import inspect


def translate_column_indices(indices, data_frame):
    return np.apply_along_axis(lambda x: data_frame.columns.values[x], 0, indices.T)


def print_stats(*args, **kwargs):
    if len(args)>0:
        print("only named arguments are accepted by this print_stats")
    for k,v in kwargs.items():
       print("Shape of {} : {}".format(k, v.shape))

