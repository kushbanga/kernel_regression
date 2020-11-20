import numpy as np

from .toeplitz import create_toeplitz_features
from .utils import Bunch

def postprocess_bhat(bhat, kernels, time_diff, intercept=True):
    """
    Transform kernels back at the end of the regression
    :param bhat: Transformed kernels
    :param kernels: Information on kernels
    :param time_diff: Indicates the type of transformation
    :param intercept: Was an intercept used
    :return: Rotated kernels
    """
    if time_diff == 0:
        return bhat

    elif time_diff == 1:
        features = create_toeplitz_features(kernels, return_size=False, intercept=intercept)
        new_bhat = np.zeros(bhat.shape)
        if intercept:
            new_bhat[0] = bhat[0]
        for name, kernel in kernels.items():
            feature = features[name]
            if 'type' not in feature.keys():
                for i in range(feature.reps):
                    new_bhat[feature.offset+i*feature.length:feature.offset+(i+1)*feature.length] =\
                        np.cumsum(bhat[feature.offset+i*feature.length:feature.offset+(i+1)*feature.length], axis=1)
        return bhat

    else:
        raise NotImplementedError('Coefficient transformation not implemented')

def get_coefs(bhat, kernels, intercept=True):
    """
    Convert coefficients into a more readable dictionary
    :param bhat: Coefficients
    :param kernels: Information on kernels
    :param intercept: Was an intercept used
    :return: Dictionary of coefficients
    """
    coefs = Bunch()
    features = create_toeplitz_features(kernels, return_size=False, intercept=intercept)
    if intercept:
        coefs.intercepts = bhat[0]
    for name, kernel in kernels.items():
        feature = features[name]
        if 'type' not in feature.keys():
            for i in range(feature.reps):
                coefs[name + str(feature.unique_types[i])] = \
                    bhat[feature.offset+i*feature.length:feature.offset+(i+1)*feature.length]
    return coefs
