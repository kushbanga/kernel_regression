import numpy as np

from .utils import Bunch, _copy


def preprocess_kernels(kernels_orig, time_bin):
    """
    Preprocess the kernels so they're ready for regression. Primarily this involves grouping events
    into time bins.
    :param kernels_orig: Kernels created from the data
    :param time_bin: Length of the time bins
    :return: Processed kernels object
    """
    kernels = Bunch()
    for key, data in kernels_orig.items():
        kernels[key] = Bunch()
        kernels[key].times = (kernels_orig[key].times // time_bin).astype(int)
        kernels[key].types = _copy(kernels_orig[key].types)
        kernels[key].interval = (np.array(kernels_orig[key].interval) // time_bin).astype(int)
        # kernels[key].exclude = _copy(kernels_orig[key].exclude)
        try:
            kernels[key].unique_types = _copy(kernels_orig[key].unique_types)
        except AttributeError:
            kernels[key].unique_types = np.unique(kernels_orig[key].types)
        try:
            kernels[key].amps = _copy(kernels_orig[key].amps)
        except AttributeError:
            kernels[key].amps = None

    return kernels
