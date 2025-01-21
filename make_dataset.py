import numpy as np
import os
import pickle

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def make_dataset(filename=None, num_samples=100, offset=0):

    def make_instance(args):
        loc, prize, depot, max_length, *args = args
        grid_size = 1
        if len(args) > 0:
            depot_types, customer_types, grid_size = args
        return {
            'loc': np.array(loc, dtype=np.float) / grid_size,
            'prize': np.array(prize, dtype=np.float),
            'depot': np.array(depot, dtype=np.float) / grid_size,
            'max_length': np.array(max_length, dtype=np.float)/ grid_size}

    if filename is not None:
        assert os.path.splitext(filename)[1] == '.pkl', 'file name error'
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        data = [make_instance(args) for args in data[offset:offset + num_samples]]
    else:
        print('filename is None')
    return data