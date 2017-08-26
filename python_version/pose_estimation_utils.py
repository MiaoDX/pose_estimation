#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some utils functions to ease our life
"""

import cv2
import numpy as np

from linear_algebra_helper import cheirality_check


def R2yzx(R):
    """
    http://www.cnblogs.com/singlex/p/RotateMatrix2Euler.html
    :param R:
    :return: thetaz, thetay, thetax
    """
    r11 = R[0][0]
    r21 = R[1][0]
    r31 = R[2][0]
    r32 = R[2][1]
    r33 = R[2][2]

    from math import pi, atan2, sqrt
    z = atan2(r21, r11) / pi * 180
    y = atan2(-r31, sqrt(r32 * r32 + r33 * r33)) / pi * 180
    x = atan2(r32, r33) / pi * 180

    return np.array([z, y, x]).reshape(3, 1)    # to make it the same as t


def Rs2zyxs(Rs):
    zyxs = []
    for R in Rs:
        zyxs.append(R2yzx(R))
    zyxs = np.array(zyxs)
    assert zyxs[0].shape == (3, 1)

    return zyxs


def DEBUG_Rt(R, t, name=""):
    print(name)

    r, _ = cv2.Rodrigues(R)
    print("R:\n{}".format(R))
    print("r:\n{}".format(r))
    z, y, x = R2yzx(R)
    print("rotate_angle:\nz:{}\ny:{}\nx:{}".format(z, y, x))
    print("t:\n{}".format(t))


def key_points_to_matched_pixel_points(first_key_points, second_key_points,
                                       matches):

    first_match_points = np.zeros((len(matches), 2), dtype=np.float32)
    second_match_points = np.zeros_like(first_match_points)
    distances = np.zeros_like(first_match_points)

    for i in range(len(matches)):
        first_match_points[i] = first_key_points[matches[i].queryIdx].pt
        second_match_points[i] = second_key_points[matches[i].trainIdx].pt

    return first_match_points, second_match_points, distances


def find_F_and_matches(kps1, kps2, matches):

    pts1, pts2, _ = key_points_to_matched_pixel_points(kps1, kps2, matches)

    F, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # We select only inlier points
    # pts1_F = pts1[mask_F.ravel() == 1]
    # pts2_F = pts2[mask_F.ravel() == 1]

    matches_F = []
    matches_F_bad = []
    for i in range(len(matches)):
        if mask_F[i] != 0:
            assert mask_F[i] == 1
            matches_F.append(matches[i])
        else:
            matches_F_bad.append(matches[i])

    print("In find_F_and_refineMatches, matches:{} -> {}".format(
        len(matches), len(matches_F)))

    return F, matches_F, matches_F_bad


def find_E_and_matches_cv2(kp1, kp2, matches, K):
    F, matches_F, matches_F_bad = find_F_and_matches(kp1, kp2, matches)
    E = K.T.dot(F).dot(K)

    return E, matches_F, matches_F_bad


def find_E_and_matches_cv3(kp1, kp2, matches, K):

    pts1, pts2, _ = key_points_to_matched_pixel_points(kp1, kp2, matches)
    """ findEssentialMat(points1, points2, cameraMatrix[, method[, prob[, threshold[, mask]]]]) -> retval, mask """
    #E, mask_E = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=0.2)
    E, mask_E = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K)

    # We select only inlier points

    matches_E = []
    matches_E_bad = []

    for i in range(len(matches)):
        if mask_E[i] != 0:
            assert mask_E[i] == 1
            matches_E.append(matches[i])
        else:
            matches_E_bad.append(matches[i])

    print(
        "In find_E_cv3, matches:{} -> {}".format(len(matches), len(matches_E)))

    return E, matches_E, matches_E_bad


def recoverPose_from_E_cv3(E, kps1, kps2, matches, K):

    pts1, pts2, _ = key_points_to_matched_pixel_points(kps1, kps2, matches)

    _, R, t, mask_rp = cv2.recoverPose(
        E, pts1, pts2,
        K)    # this can have the determiate results, so we choose it

    # We select only inlier points
    # pts1_rp = pts1[mask_rp.ravel() != 0]
    # pts2_rp = pts2[mask_rp.ravel() != 0]
    matches_rp = []
    matches_rp_bad = []
    for i in range(len(matches)):
        if mask_rp[i] != 0:
            matches_rp.append(matches[i])
        else:
            matches_rp_bad.append(matches[i])

    print("In recoverPoseFromE_cv3, points:{} -> inliner:{}".format(
        len(matches), len(matches_rp)))

    return R, t, matches_rp, matches_rp_bad


def points_pixel_to_camera(pts1, pts2, K_inv):
    assert len(pts1) == len(pts2)

    pts1_cam = []
    pts2_cam = []

    for i in range(len(pts1)):
        # normalize and homogenize the image coordinates
        pts1_cam.append(K_inv.dot([pts1[i][0], pts1[i][1], 1.0]))
        pts2_cam.append(K_inv.dot([pts2[i][0], pts2[i][1], 1.0]))

    return pts1_cam, pts2_cam


def recoverPose_from_E_cv2(E, kps1, kps2, matches, K):

    # iterate over all point correspondences used in the estimation of the
    # fundamental matrix

    pts1, pts2, _ = key_points_to_matched_pixel_points(kps1, kps2, matches)
    K_inv = np.linalg.inv(K)

    pts1_cam, pts2_cam = points_pixel_to_camera(pts1, pts2, K_inv)

    match_inliers1, match_inliers2, Rt1, Rt2 = cheirality_check(
        E, pts1_cam, pts2_cam)

    R = Rt2[:, :-1]
    t = Rt2[:, -1]

    return R, t, matches, list()
