import numpy as np


class Bunch(dict):
    """A subclass of dictionary with an additional dot syntax."""

    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self


def _copy(array):
    """
    Wrapper around Numpy's copy method to handle some edge cases
    :param array: Array to be copied
    :return: Copy of array
    """
    if array is None:
        return None
    if type(array) == np.ndarray:
        return array.copy()
