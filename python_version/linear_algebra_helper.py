"""
Copy from original scene3D
"""

import numpy as np
import cv2

def cheirality_check(E, first_inliers, second_inliers):

    """Finds the [R|t] camera matrix"""
    # decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
    U, S, Vt = np.linalg.svd(E)
    W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                  1.0]).reshape(3, 3)

    # Determine the correct choice of second camera matrix
    # only in one of the four configurations will all the points be in
    # front of both cameras
    # First choice: R = U * Wt * Vt, T = +u_3 (See Hartley Zisserman 9.19)
    R = U.dot(W).dot(Vt)
    T = U[:, 2]
    if not _in_front_of_both_cameras(first_inliers, second_inliers,
                                          R, T):
        # Second choice: R = U * W * Vt, T = -u_3
        T = - U[:, 2]

    if not _in_front_of_both_cameras(first_inliers, second_inliers,
                                          R, T):
        # Third choice: R = U * Wt * Vt, T = u_3
        R = U.dot(W.T).dot(Vt)
        T = U[:, 2]

        if not _in_front_of_both_cameras(first_inliers,
                                              second_inliers, R, T):
            # Fourth choice: R = U * Wt * Vt, T = -u_3
            T = - U[:, 2]

    match_inliers1 = first_inliers
    match_inliers2 = second_inliers
    Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    Rt2 = np.hstack((R, T.reshape(3, 1)))

    return match_inliers1, match_inliers2, Rt1, Rt2


def _in_front_of_both_cameras(first_points, second_points, rot, trans):
    """Determines whether point correspondences are in front of both
       images"""
    rot_inv = rot
    for first, second in zip(first_points, second_points):
        first_z = np.dot(rot[0, :] - second[0]*rot[2, :],
                         trans) / np.dot(rot[0, :] - second[0]*rot[2, :],
                                         second)
        first_3d_point = np.array([first[0] * first_z,
                                   second[0] * first_z, first_z])
        second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T,
                                                                 trans)

        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            return False

    return True

def _linear_ls_triangulation(u1, P1, u2, P2):
    """Triangulation via Linear-LS method"""
    # build A matrix for homogeneous equation system Ax=0
    # assume X = (x,y,z,1) for Linear-LS method
    # which turns it into AX=B system, where A is 4x3, X is 3x1 & B is 4x1
    A = np.array([u1[0]*P1[2, 0] - P1[0, 0], u1[0]*P1[2, 1] - P1[0, 1],
                  u1[0]*P1[2, 2] - P1[0, 2], u1[1]*P1[2, 0] - P1[1, 0],
                  u1[1]*P1[2, 1] - P1[1, 1], u1[1]*P1[2, 2] - P1[1, 2],
                  u2[0]*P2[2, 0] - P2[0, 0], u2[0]*P2[2, 1] - P2[0, 1],
                  u2[0]*P2[2, 2] - P2[0, 2], u2[1]*P2[2, 0] - P2[1, 0],
                  u2[1]*P2[2, 1] - P2[1, 1],
                  u2[1]*P2[2, 2] - P2[1, 2]]).reshape(4, 3)

    B = np.array([-(u1[0]*P1[2, 3] - P1[0, 3]),
                  -(u1[1]*P1[2, 3] - P1[1, 3]),
                  -(u2[0]*P2[2, 3] - P2[0, 3]),
                  -(u2[1]*P2[2, 3] - P2[1, 3])]).reshape(4, 1)

    ret, X = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
    return X.reshape(1, 3)
