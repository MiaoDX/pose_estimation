"""
Some common helper function, with no specific belongings
"""

import cv2
import numpy as np


def isMatSame(m1, m2):
    """
    is the mat the same? # https://stackoverflow.com/a/4550775/7067150
    :param m1:
    :param m2:
    :return:
    >>> arr1 = np.array(range(10))
    >>> arr2 = np.array(range(1,11))
    >>> isMatSame(arr1, arr1)
    True
    >>> isMatSame(arr1, arr2)
    False
    """
    m1 = np.array(m1)
    m2 = np.array(m2)
    diff = cv2.compare(m1, m2, cv2.CMP_NE)
    return cv2.countNonZero(diff) == 0


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
