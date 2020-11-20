import numpy as np


def get_Y(spike_times, spike_clusters, tbin):
    """
    Function to bin spike times efficiently, based on an IBL function
    :param spike_times:
    :param spike_clusters:
    :param tbin: Size of time bins
    :return:
    """
    xlim = [0, np.max(spike_times)]
    ylim = [0, np.max(spike_clusters)]

    xscale, xind = _get_scale_and_indices(spike_times, tbin, xlim)
    yscale, yind = _get_scale_and_indices(spike_clusters, 1, ylim)

    nx, ny = [xscale.size, yscale.size]
    ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
    Y = np.bincount(ind2d, minlength=nx * ny).reshape(ny, nx)

    return Y


def _get_scale_and_indices(v, bin, lim):
    # if bin is a nonzero scalar, this is a bin size: create scale and indices
    if np.isscalar(bin) and bin != 0:
        scale = np.arange(lim[0], lim[1] + bin / 2, bin)
        ind = (np.floor((v - lim[0]) / bin)).astype(np.int64)
    # if bin == 0, aggregate over unique values
    else:
        scale, ind = np.unique(v, return_inverse=True)
    return scale, ind
