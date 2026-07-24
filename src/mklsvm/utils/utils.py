import numpy as np
import datetime as dt


def train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=None):
    X = np.asarray(X)
    Y = (np.asarray(Y)).ravel()
    # --- Checking for same length ---
    if len(X) != len(Y):
        raise ValueError("X and y must have the same number of samples")
    # --- Calculating Test-Array Size ---
    n_samples = len(X)
    test_size = int(n_samples * test_size)
    indices = np.arange(n_samples)
    # --- If Shuffle is True ---
    if shuffle:
        shuffle_rng = np.random.default_rng(random_state)
        shuffle_rng.shuffle(indices)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    # Making the Final Arrays Ready ---
    x_train = X[train_idx]
    x_test  = X[test_idx]
    y_train = Y[train_idx]
    y_test  = Y[test_idx]
    # --- Return ---
    return x_train, x_test, y_train, y_test



class timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = dt.datetime.now()

    def stop(self):
        self.end_time = dt.datetime.now()

    def elapsed_time(self):
        if self.start_time is None or self.end_time is None:
            raise ValueError("Error: Timer has not been started and stopped properly.")
        return self.end_time - self.start_time