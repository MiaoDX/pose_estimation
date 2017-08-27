#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The image color transfer
"""

import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from enum import Enum    # in OpenCV 2.x, `pip install enum34` https://www.gitbook.com/book/taizilongxu/stackoverflow-about-python/details

IM_COLOR = Enum('IM_COLOR', 'IM_GRAY IM_BGR IM_BGRA')

import matplotlib.cbook as cbook
hopper_file = cbook.get_sample_data(
    'grace_hopper.png').name    # 512*600, shape (600, 512, 3)
hopper_BGR = cv2.imread(hopper_file)
hopper_GRAY = cv2.cvtColor(hopper_BGR, cv2.COLOR_BGR2GRAY)
hopper_BGRA = cv2.cvtColor(hopper_BGR, cv2.COLOR_BGR2BGRA)


def im_dim(im):
    """
    :param im:
    :return:
    >>> im_dim(hopper_GRAY)
    2
    >>> im_dim(hopper_BGR)
    3
    >>> im_dim(hopper_BGRA)
    4
    """
    assert im is not None
    im_shape = im.shape

    if len(im_shape) == 2:    # gray
        return 2
    else:
        return im_shape[-1]


def im_color(im):
    """
    :param im:
    :return:
    >>> im_color(hopper_GRAY) == IM_COLOR.IM_GRAY
    True
    >>> im_color(hopper_BGR) == IM_COLOR.IM_BGR
    True
    >>> im_color(hopper_BGRA) == IM_COLOR.IM_BGRA
    True
    """
    assert im is not None
    dim = im_dim(im)
    if dim == 2:
        return IM_COLOR.IM_GRAY
    if dim == 3:
        return IM_COLOR.IM_BGR
    if dim == 4:
        return IM_COLOR.IM_BGRA


def im2BGR_dim3(im):
    """

    :param im:
    :return:
    >>> im = im2BGR_dim3(hopper_GRAY)
    >>> im.shape[-1] == 3
    True
    >>> im = im2BGR_dim3(hopper_BGR)
    >>> im.shape[-1] == 3
    True
    >>> im = im2BGR_dim3(hopper_BGRA)
    >>> im.shape[-1] == 3
    True
    """
    assert im is not None
    im_c = im_color(im)
    if im_c == IM_COLOR.IM_GRAY:
        return np.dstack([im, im, im])
    if im_c == IM_COLOR.IM_BGR:
        return im
    if im_c == IM_COLOR.IM_BGRA:
        return cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)


def concatenate_im(im1, im2):
    """
    :param im1:
    :param im2:
    :return:
    >>> im = concatenate_im(hopper_GRAY, hopper_GRAY)
    >>> im.shape
    (600, 1024, 3)
    >>> im = concatenate_im(hopper_GRAY, hopper_BGR)
    >>> im.shape
    (600, 1024, 3)
    >>> im = concatenate_im(hopper_GRAY, hopper_BGRA)
    >>> im.shape
    (600, 1024, 3)
    """
    assert im1 is not None
    assert im2 is not None
    im1 = im2BGR_dim3(im1)
    im2 = im2BGR_dim3(im2)
    assert im1.shape == im2.shape
    return np.hstack([im1, im2])


def imshow_cv_plt(im, name="", use_plt=False):
    assert im is not None
    im = im2BGR_dim3(im)

    if not use_plt:
        cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL |
                        cv2.WINDOW_KEEPRATIO)    # not sure which is better
        cv2.imshow(name, im)
        cv2.waitKey(0)
        return

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im), plt.title(name), plt.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    """
    base_dir = "H:/projects/SLAM/python_code/dataset/our/trajs2/"

    im1_file = base_dir + '1.jpg'
    im2_file = base_dir + '7b.jpg'

    im1 = cv2.imread(im1_file)
    im2 = cv2.imread(im2_file)

    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    assert im_color(im1) == IM_COLOR.IM_BGR
    assert im_color(im2) == IM_COLOR.IM_BGR

    assert im_color(im1_gray) == IM_COLOR.IM_GRAY
    assert im_color(im1_gray) == IM_COLOR.IM_GRAY

    imshow_cv_plt(im1, "im1")
    imshow_cv_plt(im1, "im1 plt", True)

    im1_im2 = concatenate_im(im1, im2)
    imshow_cv_plt(im1_im2, "im1_im2")
    imshow_cv_plt(im1_im2, "im1_im2", True)

    im1_im2_gray = concatenate_im(im1, im2_gray)
    imshow_cv_plt(im1_im2_gray, "im1_im2_gray")
    imshow_cv_plt(im1_im2_gray, "im1_im2_gray", True)
    """
