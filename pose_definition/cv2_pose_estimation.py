"""
Demo for OpenCV pose estimation methods, aiming at point out the meaning of R,t

All APIs are constant, a.k.a:
POSE=R, t = METHOD(A, B) will bring A to B -> P_B = POSE * P_A


Please note that, if the cube chooses different params, the codes may just broken
"""

from utils.cube import cube_points, cube_points9, make_homog, convert_to_homog
from utils.POSE3 import Pose3, rotm2zyxEulDegree
import numpy as np
import cv2


if __name__ == '__main__':

    np.set_printoptions(precision=2)

    #box = cube_points([0, 0, 20], 2)
    box = cube_points9([0, 0, 20], 2)
    print(box.shape)
    # print(box)

    c6d_A = [0, 0, 0, 0, 0, 0] # (cx, cy, cz, roll, yaw, pitch)
    # c6d_A = [-5, 10, -15, -5, 10, -15]
    pose_A = Pose3().fromCenter6D(c6d_A)
    K_A = np.array([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]])

    c6d_B = [10, 0, 0, 0, 0, 0]
    # c6d_B = [10, 0, 0, 0, 45, 0]
    # c6d_B = [10, 0, 0, 0, 90, 0]
    # c6d_B = [5, -10, 15, 5, -10, 15]
    pose_B = Pose3().fromCenter6D(c6d_B)
    K_B = np.array([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]])


    box_camA = pose_A.apply_points(box)
    box_camA_pixel_3d = K_A @ box_camA
    box_camA_pixel_2d = convert_to_homog(box_camA_pixel_3d)[:2]
    print(box_camA_pixel_2d.shape)
    box_camB = pose_B.apply_points(box)
    box_camB_pixel_3d = K_B @ box_camB
    box_camB_pixel_2d = convert_to_homog(box_camB_pixel_3d)[:2]
    print(box_camB_pixel_2d.shape)



    print(box)
    print(box_camB)
    print(box_camB_pixel_2d)

    box_camB_inverse = np.linalg.inv(K_B) @ make_homog(box_camB_pixel_2d)
    print(box_camB_inverse)
    box_camB_depth = box_camB[2]
    #box_camB_back = box_camB_inverse * box_camB_depth
    box_camB_back = np.multiply(box_camB_inverse, box_camB_depth)
    print(box_camB_back)
    exit()



    """
    Homography, Finds a perspective transformation between two **planes**.
    We use first four pts, the are at one plane
    
    s_i [p2;1] ~ H [p1;1], s_i is scale
    """
    print("="*100)
    print("findHomography")
    plane_camA_2d = box_camA_pixel_2d[:, :4]
    plane_camB_2d = box_camB_pixel_2d[:, :4]

    assert plane_camA_2d.shape == (2, 4)

    # findHomography(srcPoints, dstPoints[, method[, ransacReprojThreshold[, mask[, maxIters[, confidence]]]]]) -> retval, mask
    H, status = cv2.findHomography(plane_camB_2d.T, plane_camA_2d.T, method=cv2.RANSAC)
    # print(H)

    # H back
    plane_camA_by_H = H @ make_homog(plane_camB_2d)
    plane_camA_by_H_2d = convert_to_homog(plane_camA_by_H)[:2]
    assert np.allclose(plane_camA_2d, plane_camA_by_H_2d)


    """
    Fundamental, $[p2;1]^T F [p1;1]=0$
    """
    print("="*100)
    print("findFundamentalMat")
    # findFundamentalMat(points1, points2[, method[, param1[, param2[, mask]]]]) -> retval, mask
    F, mask = cv2.findFundamentalMat(points1=box_camB_pixel_2d.T, points2=box_camA_pixel_2d.T, method=cv2.FM_RANSAC)
    # print(F)

    F_v = [p2.T @ F @ p1 for p1, p2 in zip(make_homog(box_camB_pixel_2d).T, make_homog(box_camA_pixel_2d).T)]
    print(np.max(F_v))
    #assert np.allclose(np.zeros(len(F_v)), F_v, atol=0.02)
    assert np.allclose(np.zeros(len(F_v)), F_v, atol=1e-5)

    """
    Essential, 
    $E = K.t() * F * K$ -> $F = K^{-T} E K^{-1}$ 
    -> $[p2;1]^T K^{-T} E K^{-1} [p1;1]=0$
    """
    print("="*100)
    print("findEssentialMat")

    # findEssentialMat(points1, points2, cameraMatrix[, method[, prob[, threshold[, mask]]]]) -> retval, mask
    E, mask = cv2.findEssentialMat(points1=box_camB_pixel_2d .T, points2=box_camA_pixel_2d .T, cameraMatrix=K_B, method=cv2.RANSAC)
    # print(E)
    # print(mask)

    E_v = [ p2.T @ np.linalg.inv(K_B.T) @ E @ np.linalg.inv(K_B) @ p1 for p1, p2 in
           zip(make_homog(box_camB_pixel_2d).T, make_homog(box_camA_pixel_2d).T)]
    print(max(E_v))
    assert np.allclose(np.zeros(len(E_v)), E_v, atol=1.e-5)


    """
    Recoverpose
    """
    print("="*100)
    print("recoverPose")
    # recoverPose(E, points1, points2, cameraMatrix[, R[, t[, mask]]]) -> retval, R, t, mask
    ret, R, t, mask = cv2.recoverPose(E, points1=box_camB_pixel_2d.T, points2=box_camA_pixel_2d.T, cameraMatrix=K_B)
    print(rotm2zyxEulDegree(R))
    print(rotm2zyxEulDegree(R.T))
    print(t)



    """
    PNP and pose
    """
    print("="*100)
    print("solvePnPRansac")
    # solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, iterationsCount[, reprojectionError[, confidence[, inliers[, flags]]]]]]]]) -> retval, rvec, tvec, inliers

    retval, rvec, tvec, inlier = cv2.solvePnPRansac(objectPoints=box_camB.T, imagePoints=box_camA_pixel_2d.T, cameraMatrix=K_B, distCoeffs=None, flags=cv2.SOLVEPNP_EPNP)
    R, jacobian = cv2.Rodrigues(rvec)
    print(rotm2zyxEulDegree(R))
    print(rotm2zyxEulDegree(R.T))
    print(tvec)
    print(tvec/np.linalg.norm(tvec))
    # pose = Pose3.fromRt(R, tvec)
    # # pose must be inversed
    # pose = pose.inverse()


    print("="*100)
    print("Pose directly")
    rp = Pose3().fromRt(R, tvec)
    rp.compose(pose_B).debug()

    pose_A.debug()