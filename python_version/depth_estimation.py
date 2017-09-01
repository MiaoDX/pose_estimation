"""
Code for estimating the depth

=============================
Method 1
Since our platform can perform translation with nice precision, we can use that values directly (with some loss).
Say we move right 4cm, and use rotate angle=(0.0, 0.0, 0.0), t = (40, 0, 0) and try to get the 4d points.
Then we use the 4d points to get the R,t with dimension by solvePnP
=============================
Method 2
We use the R,t we calculated from the findEssentialMat and recoverPose, and scale the t vector according to the 4cm.
=============================
"""

import cv2
import numpy as np
import pose_estimation_utils as pe_utils
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def get_and_plot_point_cloud(first_key_points, second_key_points, matches,
                             K_inv, Rt1, Rt2):
    """Plots 3D point cloud
        This method generates and plots a 3D point cloud of the recovered
        3D scene.
    """
    # Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    # Rt2 = np.hstack((R, t.reshape(3, 1)))
    assert Rt1.shape == (3, 4) and Rt2.shape == (3, 4)

    # get the points, in camera coordinate
    pts1, pts2, _ = pe_utils.key_points_to_matched_pixel_points(
        first_key_points, second_key_points, matches)

    pts1_cam, pts2_cam = pe_utils.points_pixel_to_camera(pts1, pts2, K_inv)

    # triangulate points
    first_inliers_2xN = pts1_cam[:, :2].T
    second_inliers_2xN = pts2_cam[:, :2].T

    pts4D_Nx4 = cv2.triangulatePoints(Rt1, Rt2, first_inliers_2xN,
                                      second_inliers_2xN).T

    # convert from homogeneous coordinates to 3D
    pts3D_Nx3 = pts4D_Nx4[:, :3] / np.repeat(pts4D_Nx4[:, 3], 3).reshape(-1, 3)

    # plot with matplotlib
    Xs = pts3D_Nx3[:, 0]
    Ys = pts3D_Nx3[:, 1]
    Zs = pts3D_Nx3[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # coord = {'minX': -150.0, 'minY': -100.0, 'minZ': 350.0, 'maxZ': 450.0}
    # ax.set_xlim((coord["minX"], -coord["minX"]))
    # ax.set_ylim((coord["minZ"], coord["maxZ"]))
    # ax.set_zlim((coord["minY"], -coord["minY"]))

    # ax.scatter(Xs, Ys, Zs, c='r', marker='o')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    ax.scatter(Xs, Zs, Ys, c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    ax.invert_zaxis()

    plt.title('3D point cloud: Use pan axes button below to inspect')
    plt.show()

    return pts3D_Nx3
