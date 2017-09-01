"""
Some code for refining various values
"""

import cv2
import pose_estimation_utils as pe_utils
import numpy as np


def correctMatches_with_E(E, K, kps1, kps2, matches):
    """
    I am not so sure how to use this function, since it changes the coordinately of the points :<
    [ref](http://answers.opencv.org/question/341/python-correctmatches/?answer=402#post-id-402)
    :param E:
    :param K:
    :param kps1:
    :param kps2:
    :param matches:
    :return:
    """
    F_backward = pe_utils.find_F_from_E_and_K(E, K)
    """correctMatches(F, points1, points2[, newPoints1[, newPoints2]]) -> newPoints1, newPoints2"""

    pts1, pts2, _ = pe_utils.key_points_to_matched_pixel_points(
        kps1, kps2, matches)

    for i in range(10):
        print(pts1[i])

    pts1_tmp = np.reshape(pts1, (1, -1, 2))
    pts2_tmp = np.reshape(pts2, (1, -1, 2))

    newPts1, newPts2 = cv2.correctMatches(F_backward, pts1_tmp, pts2_tmp)

    newPts1 = newPts1.reshape(-1, 2)
    newPts2 = newPts2.reshape(-1, 2)

    for i in range(10):
        print(newPts1[i])

    print(
        "In correctMatches: pts1.shape:{}->newPts1.shape:{}, newPts2.shape:{}".
        format(pts1.shape, newPts1.shape, newPts2.shape))
