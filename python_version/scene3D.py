#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module that contains an algorithm for 3D scene reconstruction """

import cv2
import numpy as np
import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import imutils
import pose_estimation_utils as pe_utils
import image_color_utils as im_cutils

DEBUG = True

class SceneReconstruction3D:
    """3D scene reconstruction

        This class implements an algorithm for 3D scene reconstruction using
        stereo vision and structure-from-motion techniques.

        A 3D scene is reconstructed from a pair of images that show the same
        real-world scene from two different viewpoints. Feature matching is
        performed either with rich feature descriptors or based on optic flow.
        3D coordinates are obtained via triangulation.

        Note that a complete structure-from-motion pipeline typically includes
        bundle adjustment and geometry fitting, which are out of scope for
        this project.
    """
    def __init__(self, K, dist):
        """Constructor

            This method initializes the scene reconstruction algorithm.

            :param K: 3x3 intrinsic camera matrix
            :param dist: vector of distortion coefficients
        """
        self.K = K
        self.K_inv = np.linalg.inv(K)  # store inverse for fast access
        self.d = dist

    def load_image_pair(self, img_path1, img_path2, use_pyr_down=False, target_width = 600):
        """Loads pair of images

            This method loads the two images for which the 3D scene should be
            reconstructed. The two images should show the same real-world scene
            from two different viewpoints.

            :param img_path1: path to first image
            :param img_path2: path to second image
            :param use_pyr_down: flag whether to downscale the images to
                                 roughly 600px width (True) or not (False)
        """
        self.img1_color = cv2.imread(img_path1)
        self.img2_color = cv2.imread(img_path2)

        self.img1 = cv2.cvtColor(self.img1_color, cv2.COLOR_BGR2GRAY)
        self.img2 = cv2.cvtColor(self.img2_color, cv2.COLOR_BGR2GRAY)

        # make sure images are valid
        if self.img1 is None:
            sys.exit("Image " + img_path1 + " could not be loaded.")
        if self.img2 is None:
            sys.exit("Image " + img_path2 + " could not be loaded.")


        # scale down image if necessary
        # to something close to target_width wide (px)
        if use_pyr_down and self.img1.shape[1] > target_width:
            while self.img1.shape[1] > 2*target_width:
                self.img1 = cv2.pyrDown(self.img1)
                self.img2 = cv2.pyrDown(self.img2)

        # undistort the images
        self.img1 = cv2.undistort(self.img1, self.K, self.d)
        self.img2 = cv2.undistort(self.img2, self.K, self.d)

        img1_color_img2_color = im_cutils.concatenate_im(self.img1_color, self.img2_color)
        im_cutils.imshow_cv_plt(img1_color_img2_color, "The image pairs", True)

    def plot_optic_flow(self):
        """Plots optic flow field

            This method plots the optic flow between the first and second
            image.
        """
        self._extract_keypoints("flow")

        img = self.img1
        for i in range(len(self.match_pts1)):
            cv2.line(img, tuple(self.match_pts1[i]), tuple(self.match_pts2[i]),
                     color=(255, 0, 0))
            theta = np.arctan2(self.match_pts2[i][1] - self.match_pts1[i][1],
                               self.match_pts2[i][0] - self.match_pts1[i][0])
            cv2.line(img, tuple(self.match_pts2[i]),
                     (np.int(self.match_pts2[i][0] - 6*np.cos(theta+np.pi/4)),
                      np.int(self.match_pts2[i][1] - 6*np.sin(theta+np.pi/4))),
                     color=(255, 0, 0))
            cv2.line(img, tuple(self.match_pts2[i]),
                     (np.int(self.match_pts2[i][0] - 6*np.cos(theta-np.pi/4)),
                      np.int(self.match_pts2[i][1] - 6*np.sin(theta-np.pi/4))),
                     color=(255, 0, 0))

        im_cutils.imshow_cv_plt(img, "Image Flow", True)

    def draw_epipolar_lines(self, feat_mode="SURF"):
        """Draws epipolar lines

            This method computes and draws the epipolar lines of the two
            loaded images.

            :param feat_mode: whether to use rich descriptors for feature
                              matching ("surf") or optic flow ("flow")
        """
        self._extract_keypoints(feat_mode)
        self._find_fundamental_matrix()
        # Find epilines corresponding to points in right image (second image)
        # and drawing its lines on left image
        pts2re = self.match_pts2.reshape(-1, 1, 2)
        lines1 = cv2.computeCorrespondEpilines(pts2re, 2, self.F)
        lines1 = lines1.reshape(-1, 3)
        img3, img4 = self._draw_epipolar_lines_helper(self.img1, self.img2,
                                                      lines1, self.match_pts1,
                                                      self.match_pts2)

        img3_img4 = im_cutils.concatenate_im(img3, img4)
        im_cutils.imshow_cv_plt(img3_img4, "Find in right, draw in left", True)

        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        pts1re = self.match_pts1.reshape(-1, 1, 2)
        lines2 = cv2.computeCorrespondEpilines(pts1re, 1, self.F)
        lines2 = lines2.reshape(-1, 3)
        img1, img2 = self._draw_epipolar_lines_helper(self.img2, self.img1,
                                                      lines2, self.match_pts2,
                                                      self.match_pts1)

        img1_img2 = im_cutils.concatenate_im(img1, img2)
        im_cutils.imshow_cv_plt(img1_img2, "Find in left, draw in right", True)


    def plot_rectified_images(self, feat_mode="SURF"):
        """Plots rectified images

            This method computes and plots a rectified version of the two
            images side by side.

            :param feat_mode: whether to use rich descriptors for feature
                              matching ("surf") or optic flow ("flow")
        """
        self._extract_keypoints(feat_mode)
        self._find_fundamental_matrix()
        self._find_essential_matrix()
        self._find_camera_matrices_rt()

        R = self.Rt2[:, :3]
        T = self.Rt2[:, 3]
        #perform the rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.K, self.d,
                                                          self.K, self.d,
                                                          self.img1.shape[:2],
                                                          R, T, alpha=1.0)
        mapx1, mapy1 = cv2.initUndistortRectifyMap(self.K, self.d, R1, self.K,
                                                   self.img1.shape[:2],
                                                   cv2.CV_32F)
        mapx2, mapy2 = cv2.initUndistortRectifyMap(self.K, self.d, R2, self.K,
                                                   self.img2.shape[:2],
                                                   cv2.CV_32F)
        img_rect1 = cv2.remap(self.img1, mapx1, mapy1, cv2.INTER_LINEAR)
        img_rect2 = cv2.remap(self.img2, mapx2, mapy2, cv2.INTER_LINEAR)

        # draw the images side by side
        total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
                      img_rect1.shape[1] + img_rect2.shape[1], 3)
        img = np.zeros(total_size, dtype=np.uint8)
        img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
        img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

        # draw horizontal lines every 25 px accross the side by side image
        for i in range(20, img.shape[0], 25):
            cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))

        im_cutils.imshow_cv_plt(img, "Two images after imgRectified", True)

    def plot_point_cloud(self, feat_mode="SURF"):
        """Plots 3D point cloud

            This method generates and plots a 3D point cloud of the recovered
            3D scene.

            :param feat_mode: whether to use rich descriptors for feature
                              matching ("surf") or optic flow ("flow")
        """
        self._extract_keypoints(feat_mode)
        self._find_fundamental_matrix()
        self._find_essential_matrix()
        self._find_camera_matrices_rt()

        # get the points, in camera coordinate
        pts1, pts2, _ = pe_utils.key_points_to_matched_pixel_points(self.first_key_points, self.second_key_points, self.matches_E)

        pts1_cam, pts2_cam = pe_utils.points_pixel_to_camera(pts1, pts2, self.K_inv)

        Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        Rt2 = np.hstack((self.R, self.t.reshape(3, 1)))

        # triangulate points
        first_inliers = np.array(pts1_cam).reshape(-1, 3)[:, :2]
        second_inliers = np.array(pts2_cam).reshape(-1, 3)[:, :2]
        pts4D = cv2.triangulatePoints(Rt1, Rt2, first_inliers.T,
                                      second_inliers.T).T

        # convert from homogeneous coordinates to 3D
        pts3D = pts4D[:, :3]/np.repeat(pts4D[:, 3], 3).reshape(-1, 3)

        # plot with matplotlib
        Ys = pts3D[:, 0]
        Zs = pts3D[:, 1]
        Xs = pts3D[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Xs, Ys, Zs, c='r', marker='o')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_zlabel('X')
        plt.title('3D point cloud: Use pan axes button below to inspect')
        plt.show()

    def _extract_keypoints(self, feat_mode):
        """Extracts keypoints

            This method extracts keypoints for feature matching based on
            a specified mode:
            - "surf": use rich SURF descriptor
            - "flow": use optic flow

            :param feat_mode: keypoint extraction mode ("surf" or "flow")
        """
        # extract features
        if feat_mode.lower() == "surf":
            # feature matching via SURF and BFMatcher
            self._extract_keypoints_surf()
        elif feat_mode.lower() == "orb":
            # feature matching via ORB and BFMatcher
            self._extract_keypoints_orb()
        elif feat_mode.lower() == "flow":
            # feature matching via optic flow
            self._extract_keypoints_flow()
        else:
            sys.exit("Unknown feat_mode " + feat_mode +
                     ". Use 'SURF' or 'FLOW' or 'ORB'")

        if DEBUG:
            print("We found {} pairs of points in total".format(len(self.match_pts1)))

    def _extract_keypoints_surf(self):
        """Extracts keypoints via SURF descriptors"""
        # extract keypoints and descriptors from both images
        if imutils.is_cv2():
            detector = cv2.SURF(250)
        else:
            detector = cv2.xfeatures2d.SURF_create(250)

        self.first_key_points, self.first_desc = detector.detectAndCompute(self.img1,
                                                                 None)
        self.second_key_points, self.second_desc = detector.detectAndCompute(self.img2,
                                                                   None)

        # match descriptors
        matcher = cv2.BFMatcher(cv2.NORM_L1, True)

        self.matches = matcher.match(self.first_desc, self.second_desc) # THIS IS NOT RIGHT, WE NEED USE KNNSEARCH

        self.match_pts1, self.match_pts2, _ = pe_utils.key_points_to_matched_pixel_points(self.first_key_points,
                                                                                       self.second_key_points,
                                                                                       self.matches)

    def _extract_keypoints_orb(self):
        """Extracts keypoints via ORB descriptors"""
        # extract keypoints and descriptors from both images
        if imutils.is_cv2():
            detector = cv2.ORB()
        else:
            detector = cv2.ORB_create()

        self.first_key_points, self.first_desc = detector.detectAndCompute(self.img1,
                                                                 None)
        self.second_key_points, self.second_desc = detector.detectAndCompute(self.img2,
                                                                   None)

        # create BFMatcher object
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)  # to make sure we are the same as the C++ version

        # Match descriptors.
        matches = bf.match(self.first_desc, self.second_desc)

        # Sort them in the order of their distance.
        matches_sorted = sorted(matches, key=lambda x: x.distance)

        print("Max distance:{}, min distance:{} in ORB".format(matches_sorted[-1].distance, matches_sorted[0].distance))
        # 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
        prune_dis = max(matches_sorted[0].distance, 30.0)

        self.matches = list(filter(lambda x: x.distance <= prune_dis, matches))

        self.match_pts1, self.match_pts2, _ = pe_utils.key_points_to_matched_pixel_points(self.first_key_points,
                                                                                       self.second_key_points,
                                                                                       self.matches)

    def _extract_keypoints_flow(self):
        """Extracts keypoints via optic flow"""
        # find FAST features
        if imutils.is_cv2():
            fast = cv2.FastFeatureDetector()
        else:
            fast = cv2.FastFeatureDetector_create()

        first_key_points = fast.detect(self.img1, None)

        first_key_list = [i.pt for i in first_key_points]
        first_key_arr = np.array(first_key_list).astype(np.float32)
        """ calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts[, status[, err[, winSize[, maxLevel[, criteria[, flags[, minEigThreshold]]]]]]]) -> nextPts, status, err """
        second_key_arr, status, err = cv2.calcOpticalFlowPyrLK(self.img1,
                                                               self.img2,
                                                               first_key_arr,
                                                               None)

        # filter out the points with high error
        # keep only entries with status=1 and small error
        condition = (status == 1) * (err < 5.)
        concat = np.concatenate((condition, condition), axis=1)
        first_match_points = first_key_arr[concat].reshape(-1, 2)
        second_match_points = second_key_arr[concat].reshape(-1, 2)

        self.match_pts1 = first_match_points
        self.match_pts2 = second_match_points

    def _find_fundamental_matrix(self):
        """Estimates fundamental matrix """
        self.F, self.matches_F, _ = pe_utils.find_F_and_matches(self.first_key_points, self.second_key_points, self.matches)

    def _refine_with_fundamental_matrix(self):
        """Estimates fundamental matrix """
        self.F, self.matches_F, _ = pe_utils.find_F_and_matches(self.first_key_points, self.second_key_points, self.matches)
        self.matches = self.matches_F

    def _find_essential_matrix(self):
        """Estimates essential matrix based on fundamental matrix """

        if imutils.is_cv2():
            # note, in OpenCV 2.x we use F to calc E, however, to make the API alike, we do the calculation inside
            self.E, self.matches_E, self.matches_E_bad = pe_utils.find_E_and_matches_cv2(self.first_key_points,
                                                                                         self.second_key_points,
                                                                                         self.matches,
                                                                                         self.K)
            print("E from FundamentalMat:{}".format(self.E))

        elif imutils.is_cv3():
            self.E, self.matches_E, self.matches_E_bad = pe_utils.find_E_and_matches_cv2(self.first_key_points,
                                                                                         self.second_key_points,
                                                                                         self.matches,
                                                                                         self.K)
            print("E from findEssentialMat:{}".format(self.E))


    def _find_camera_matrices_rt(self):

        if imutils.is_cv3():

            print("USE recoverPose in cv3:")

            self.R, self.t, matches_rp_cv3, matches_rp_bad_cv3 = pe_utils.recoverPose_from_E_cv3(self.E, self.first_key_points,
                                                                               self.second_key_points, self.matches_E,
                                                                               self.K)

            pe_utils.DEBUG_Rt(self.R, self.t, "R t from recoverPose")

        elif imutils.is_cv2():
            self.R, self.t, matches_rp, matches_rp_bad = pe_utils.recoverPose_from_E_cv2(self.E, self.first_key_points,
                                                                               self.second_key_points, self.matches_E,
                                                                               self.K)

            pe_utils.DEBUG_Rt(self.R, self.t, "R t from linear algebra")


    def _draw_epipolar_lines_helper(self, img1, img2, lines, pts1, pts2):
        """Helper method to draw epipolar lines and features """
        if img1.shape[2] == 1:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if img2.shape[2] == 1:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        c = img1.shape[1]
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0]*c) / r[1]])
            cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            cv2.circle(img1, tuple(pt1), 5, color, -1)
            cv2.circle(img2, tuple(pt2), 5, color, -1)
        return img1, img2

