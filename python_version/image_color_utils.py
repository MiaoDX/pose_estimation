"""
The image color transfer
"""


import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


from enum import Enum  # in OpenCV 2.x, `pip install enum34` https://www.gitbook.com/book/taizilongxu/stackoverflow-about-python/details

IM_COLOR = Enum('IM_COLOR', 'IM_GRAY IM_BGR IM_BGRA')



def im_dim(im):
    return len(im.shape)


def im_color(im):
    dim = im_dim(im)
    if dim == 2:
        return IM_COLOR.IM_GRAY
    if dim == 3:
        return IM_COLOR.IM_BGR
    if dim == 4:
        return IM_COLOR.IM_BGRA


def im2BGR_dim3(im):
    assert not im is None
    im_c = im_color(im)
    if im_c == IM_COLOR.IM_GRAY:
        return np.dstack([im, im, im])
    if im_c == IM_COLOR.IM_BGR:
        return im
    if im_c == IM_COLOR.IM_BGRA:
        return cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)


def concatenate_im(im1, im2):
    im1 = im2BGR_dim3(im1)
    im2 = im2BGR_dim3(im2)
    return np.hstack([im1, im2])


def imshow_cv_plt(im, name="", use_plt=False):

    im = im2BGR_dim3(im)

    if not use_plt:
        cv2.imshow(name, im)
        cv2.waitKey(0)
        return

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im), plt.title(name), plt.show()


if __name__ == "__main__":

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
