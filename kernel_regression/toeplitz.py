import numpy as np
import numba as nb

from .utils import Bunch


def create_toeplitz_matrix(kernels, n, time_diff=0, intercept=True):
    features, size = create_toeplitz_features(kernels, intercept=intercept)
    toep_matrix = np.zeros((n, size))

    if intercept:
        toep_matrix[:,0] = 1

    for name, kernel in kernels.items():
        feature = features[name]
        fill_toeplitz(toep_matrix, kernel.times, kernel.types, feature.offset, kernel.interval,
                      time_diff=time_diff, unique_types=kernel.unique_types, trial_amps=kernel.amps)

    return toep_matrix


def create_toeplitz_features(kernels, return_size=True, intercept=True):
    """
    Creates a dictionary which provides the feature-offset, feature-length and number of
    repetitions for each kernel.
    :param kernels: Dictionary providing relevant user-inputted information for each kernel
    :param return_size: If true, also return the required width of the Toeplitz matrix
    :param intercept: Boolean, whether or not to include an intercept
    :return: Returns a dictionary for each kernel providing necessary values to construct the
            toeplitz matrix
    """
    features = Bunch()
    if intercept:
        size = 1
    else:
        size = 0
    for name, kernel in kernels.items():
        features[name] = Bunch()
        features[name].length = kernels[name].interval[1] - kernels[name].interval[0]

        unique_types = np.unique(kernels[name].types[~np.isnan(kernels[name].types)])
        #if kernels[name].exclude:
        #unique_types = unique_types[~np.isin(unique_types, kernels[name].exclude)]
        features[name].unique_types = unique_types

        features[name].reps = len(unique_types)
        features[name].offset = size
        size += features[name].length * features[name].reps

    if return_size:
        return features, size

    return features


def fill_toeplitz(matrix, times, types, offset, interval, time_diff=0,
                  unique_types=None, trial_amps=None):
    """
    Fill the toeplitz matrix for a given kernel
    :param matrix: Matrix to be filled
    :param times: Event times (must be integer, i.e are already rounded to required precision)
    :param types: Event types
    :param offset: Where in the toeplitz matrix is this kernel located
    :param interval: Time interval of the kernel (in integer frames)
    :param time_diff: 0, 1 or 2 for 0th, 1st or 2nd time difference Toeplitz matrix
    :param unique_types: Unique values of event types, optional
    :param trial_amps: Trial amplitudes
    :return: None, matrix is filled directly
    """
    assert np.issubdtype(times.dtype, np.integer) # check times are integer values
    assert matrix.shape[0] > np.max(times) # check matrix has required space for the times
    assert len(times) == len(types)

    if trial_amps:
        assert len(trial_amps) == len(times)
    else:
        trial_amps = np.ones(len(times))

    if unique_types is None:
        unique_types = np.unique(types)

    # test if type is in set of interest unique_types
    test_mask = np.isin(types, unique_types)
    # locate index of trial type in unique_types
    unique_indices = np.searchsorted(unique_types, types)
    # set index of disallowed types to -1
    unique_indices[~test_mask] = -1


    if time_diff == 0:
        _fill_toeplitz_0(matrix, times, offset, interval, unique_indices, trial_amps)

    if time_diff == 1:
        _fill_toeplitz_1(matrix, times, offset, interval, unique_indices, trial_amps)

    if time_diff == 2:
        _fill_toeplitz_2(matrix, times, offset, interval, unique_indices, trial_amps)


@nb.njit()
def _fill_toeplitz_0(matrix, times, offset, interval, unique_indices, trial_amps):
    """
    Function for creating regular Toeplitz matrix
    :param matrix: Matrix to fill
    :param times: Event times
    :param offset: Kernels offset in toeplitz matrix
    :param interval: Time interval of interest
    :param unique_indices: Which kernel to assign each trial event to
    :param trial_amps: Trial amplitudes
    :return: None, matrix is filled directly
    """
    interval_length = interval[1] - interval[0]

    for i in nb.prange(len(times)):
        if unique_indices[i] != -1:
            trial_offsets = offset + interval_length * unique_indices[i]
            for j in nb.prange(interval_length):
                matrix[times[i] + interval[0] + j, trial_offsets + j] = trial_amps[i]


@nb.njit()
def _fill_toeplitz_1(matrix, times, offset, interval, unique_indices, trial_amps):
    """
    Function for creating Toeplitz matrix with first order time difference
    If the signal is expected to be piecewise constant this should give a sparse representation
    :param matrix: Matrix to fill
    :param times: Event times
    :param offset: Kernels offset in toeplitz matrix
    :param interval: Time interval of interest
    :param unique_indices: Which kernel to assign each trial event to
    :param trial_amps: Trial amplitudes
    :return: None, matrix is filled directly
    """
    interval_length = interval[1] - interval[0]

    for i in nb.prange(len(times)):
        if unique_indices[i] != -1:
            trial_offsets = offset + interval_length * unique_indices[i]
            for j in nb.prange(interval_length):
                for k in nb.prange(j+1):
                    matrix[times[i] + interval[0] + j, trial_offsets + k] = trial_amps[i]


@nb.njit()
def _fill_toeplitz_2(matrix, times, offset, interval, unique_indices, trial_amps):
    """
    Function for creating Toeplitz matrix with second order time difference
    If the signal is expected to be piecewise linear this should give a sparse representation
    :param matrix: Matrix to fill
    :param times: Event times
    :param offset: Kernels offset in toeplitz matrix
    :param interval: Time interval of interest
    :param unique_indices: Which kernel to assign each trial event to
    :param trial_amps: Trial amplitudes
    :return: None, matrix is filled directly
    """
    interval_length = interval[1] - interval[0]

    for i in nb.prange(len(times)):
        if unique_indices[i] != -1:
            trial_offsets = offset + interval_length * unique_indices[i]
            for j in nb.prange(interval_length):
                matrix[times[i] + interval[0] + j, trial_offsets: trial_offsets + (j + 1)] = \
                    np.arange(j+1, 0, -1) * trial_amps[i]
