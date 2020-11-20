import numpy as np
from tqdm.notebook import tqdm

from .regression import RRR, r2


def core(data, kernels, lamdas, rank_range, seed=None):
    """
    Core pipeline, WIP
    :param data: Contains spiking data and some trial information
    :param kernels: Dictionary of kernels including times and types
    :param lamdas: Vector of ridge regression paramters to optimise over
    :param rank_range: Range of reduced ranks to cross-validate over
    :param seed: Set seed for reproducible cross validation splits, seed must be positive integer
    :return:
    """

    # WIP: List of important parameters
    # time_diff = 2
    n_lamdas = len(lamdas)
    n_ranks = rank_range[1] - rank_range[0]

    outputs = np.zeros((3, n_lamdas, n_ranks))

    # Pre-process
    Y = data.Y
    X = data.X
    # X = create_toeplitz_matrix(kernels, Y.shape[0])
    trial_times = data.trial_times

    if seed is not None:
        assert type(seed) == int
        np.random.seed(seed)

    random_order = np.random.permutation(trial_times.shape[0])
    tenth = trial_times.shape[0] / 10
    train = np.concatenate([np.arange(trial_times[n, 0], trial_times[n, 1])
                            for n in random_order[:int(8 * tenth)]])
    valid = np.concatenate([np.arange(trial_times[n, 0], trial_times[n, 1])
                            for n in random_order[int(8 * tenth):int(9 * tenth)]])
    test = np.concatenate([np.arange(trial_times[n, 0], trial_times[n, 1])
                           for n in random_order[int(9 * tenth):]])

    X_train = X[train]
    X_valid = X[valid]
    X_test = X[test]

    Y_train = Y[train]
    Y_valid = Y[valid]
    Y_test = Y[test]

    var = np.zeros((3, Y.shape[1]))
    var[0] = Y_train.var(axis=0)
    var[1] = Y_valid.var(axis=0)
    var[2] = Y_test.var(axis=0)
    var_scale = var / np.sum(var, axis=1)[:, np.newaxis] * 100

    xtx = np.matmul(X_train.T, X_train)
    xty = np.matmul(X_train.T, Y_train)

    # Main Iterative Loop:
    for i in tqdm(range(n_lamdas)):

        # Regression
        bhat = np.linalg.solve(xtx + lamdas[i]*np.eye(xtx.shape[0]), xty)

        # Taking a low rank approximation

        rrr = RRR()
        rrr.fit(bhat, X_train)

        for j in range(n_ranks):

            yhat = rrr.predict(X_train, rank_range[0] + j)
            outputs[0,i,j] = np.dot(var_scale[0], r2(Y_train, yhat, var[0]))

            yhat = rrr.predict(X_valid, rank_range[0] + j)
            outputs[1,i,j] = np.dot(var_scale[1], r2(Y_valid, yhat, var[1]))

            yhat = rrr.predict(X_test, rank_range[0] + j)
            outputs[2,i,j] = np.dot(var_scale[2], r2(Y_test, yhat, var[2]))

    best_values = np.unravel_index(np.argmax(outputs[1]), outputs[1].shape)

    bhat = np.linalg.solve(xtx + lamdas[best_values[0]]*np.eye(xtx.shape[0]), xty)
    rrr = RRR()
    rrr.fit(bhat, X_train)
    best_bhat = rrr.gen_bhat(best_values[1])

    return best_bhat, outputs
