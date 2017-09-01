"""
Use perfect points to digest the functions
"""

import cv2
import numpy as np


def get_n_pts_pairs(num, K):
    pass
    bound_2d = 5


    rvec = np.array([0.1, 0.2, 0.3]).reshape(3, 1)
    tvec = np.array([0.4, 0.5, 0.6]).reshape(3, 1)

    R, _ = cv2.Rodrigues(rvec)
    tvec = cv2.normalize(tvec, None)

    # print("Expected rvec: {}".format(rvec.T))
    # print("Expected R: {}".format(R))
    # print("Expected tvec: {}".format(tvec.T))

    np.random.seed(42)

    Xs = np.random.random((num, 3))*bound_2d - bound_2d/2
    # print(Xs)
    # print(Xs.shape)


    #- camera to pixel
    x1s = K.dot(Xs.T)
    # print(x1s)

    x2s = R.dot(Xs.T)
    x2s += tvec
    # print(x2s)
    x2s = K.dot(x2s)

    x1s /= x1s[-1]
    x2s /= x2s[-1]

    # print(x1s)
    # print(x2s)
    return x1s.T[:, :2], x2s.T[:, :2]


def find_E_from_pts(pts1, pts2, K):
    assert pts1.shape[1] == 2
    assert pts2.shape[1] == 2

    print("pts1.shape:{}".format(pts1.shape))
    print("pts2.shape:{}".format(pts2.shape))

    E, mask_E = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K)

    print(E.shape)

    # We select only inlier points

    index_E = []
    index_E_bad = []

    for i in range(len(pts1)):
        if mask_E[i] != 0:
            assert mask_E[i] == 1
            index_E.append(i)
        else:
            index_E_bad.append(i)

    print(
        "In find_E_cv3, matches:{} -> {}".format(len(index_E), len(pts1)))

    return E, index_E


def recoverPose_from_E_and_pts(E, pts1, pts2, K):
    _, R, t, mask_rp = cv2.recoverPose(
        E, pts1, pts2,
        K)    # this can have the determiate results, so we choose it

    # We select only inlier points
    # pts1_rp = pts1[mask_rp.ravel() != 0]
    # pts2_rp = pts2[mask_rp.ravel() != 0]
    matches_rp = []
    matches_rp_bad = []
    for i in range(len(pts1)):
        if mask_rp[i] != 0:
            assert mask_rp[i] == 255    # this is special, pretty special
            matches_rp.append(i)
        else:
            matches_rp_bad.append(i)

    print("In recoverPoseFromE_cv3, points:{} -> inliner:{}".format(
        len(pts1), len(matches_rp)))

    return R, t


if __name__ == "__main__":

    focal = 300
    pp = (0, 0)
    K = np.array([focal, 0, pp[0], 0, focal, pp[1], 0, 0, 1]).reshape(3, 3)

    pts1, pts2 = get_n_pts_pairs(50, K)
    # print(pts1)
    # print(pts2)

    E, index_E = find_E_from_pts(pts1, pts2, K)

    print(E.shape)
    print("E:\n{}".format(E))

    R, t = recoverPose_from_E_and_pts(E, pts1, pts2, K)


    import pose_estimation_utils as pe_utils
    pe_utils.find_E_from_R_t(R, t)
    print("E from R, t:\n{}".format(E))







"""
When n is 5
the calculated E shape (24, 3), this should be a bug, right?
"""