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
import keypoints_descriptors_utils as kd_utils
import depth_estimation
import os


class Image:
    """Image
        We use a class to store the image informations
    """

    def __init__(self, img_name=None, img_color=None, img_gray=None):
        self.img_name = img_name
        self.img_color = img_color
        self.img_gray = img_gray

        self.key_points = None
        self.descriptors = None
        #self.matches = None

        # self.P = None
        self.R = None
        self.t = None
        if img_name is not None:
            self.base_name = os.path.basename(img_name).split('.')[0]

    def set_matches(self, mathces):
        self.matches = mathces

    def set_img(self, img_color, img_gray):
        self.img_color = img_color
        self.img_gray = img_gray

    def set_Rt(self, R, t):
        self.R = R
        self.t = t

    def set_key_points_and_descriptors(self, key_points, descriptors):
        self.key_points = key_points
        self.descriptors = descriptors


class CameraRelocation:
    """CameraRelocation

        This class implements an algorithm for CameraRelocation using
        stereo vision and structure-from-motion techniques.

        A 3D scene is reconstructed from a pair of images that show the same
        real-world scene from two different viewpoints. Feature matching is
        performed either with rich feature descriptors or based on optic flow.
        3D coordinates are obtained via triangulation.

        Note that a complete structure-from-motion pipeline typically includes
        bundle adjustment and geometry fitting, which are out of scope for
        this project.
    """

    def __init__(self, K, dist, feature_name="ORB", output_folder='output'):
        """Constructor

            This method initializes the scene reconstruction algorithm.

            :param K: 3x3 intrinsic camera matrix
            :param dist: vector of distortion coefficients
        """
        self.K = K
        self.K_inv = np.linalg.inv(K)    # store inverse for fast access
        self.d = dist
        self.output_folder = output_folder
        self.img1 = Image()
        self.img2 = Image()

        self.R = np.ones((3, 3))
        self.t = np.ones((3, 1))

        self.FIND_3D_POINTS = False

        # Ready to go
        self.set_feature_detector_descriptor_extractor(feature_name)
        # self.set_matcher(withFlann=False) # already in set_feature_detector_descriptor_extractor

    def forward(self, new_frame_name):

        if self.img1.img_name is None:
            print("Load the left image")
            self.load_image_left(new_frame_name)
            return

        # NOT the first one
        print("Load the right image")
        self.load_image_right(new_frame_name)

        self._get_keypoints_and_descripotrs()
        self._get_matches()
        self._refine_with_fundamental_matrix()

        if imutils.is_cv2():
            # self._find_fundamental_matrix() # already refined
            self._find_essential_matrix()
            self._find_camera_matrices_rt()
        else:
            self._refine_rt_with_ransac()
        """
        if not self.FIND_3D_POINTS:    # we have not estimated the 3d points
            self._plot_point_cloud()
            self._get_BIG_DICT_points_pixel_to_3d()    # store the relation
            self.FIND_3D_POINTS = True
        else:
            self._get_pts3D_Nx3_from_BIG_DICT()    # get the 3d points
            self._PNPSolver_img2_pts_and_3DPoints()
        """

    def load_image_left(self, img_path):
        img_color, img_gray = self._load_image(img_path)
        self.img1 = Image(img_path, img_color, img_gray)

    def load_image_right(self, img_path):
        img_color, img_gray = self._load_image(img_path)
        self.img2 = Image(img_path, img_color, img_gray)

    def _load_image(self, img_path, use_pyr_down=False, target_width=600):
        img_color = cv2.imread(img_path)
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # make sure images are valid
        if img is None:
            sys.exit("Image " + img_path + " could not be loaded.")

        # scale down image if necessary to something close to target_width wide (px)
        if use_pyr_down and img.shape[1] > target_width:
            while img.shape[1] > 2 * target_width:
                img = cv2.pyrDown(img)
        # undistort the images
        img = cv2.undistort(img, self.K, self.d)
        return img_color, img

    def set_feature_detector_descriptor_extractor(
            self,
            featurename="ORB",
            descriptor_extractor_name=None,
            feature_detector_params=None,
            descriptor_extractor_params=None):
        self.feature_detector, self.descriptor_extractor, self.normType = kd_utils.get_feature_detector_descriptor_extractor(
            featurename, descriptor_extractor_name, feature_detector_params,
            descriptor_extractor_params)

        self.set_matcher(
        )    # to make sure we won't forget by do it more than one

    def set_matcher(self, withFlann=False):
        self.matcher = kd_utils.get_matcher(self.normType, withFlann)

    def _get_keypoints_and_descripotrs(self):

        if self.img1.key_points is None:
            self.img1.key_points, self.img1.descriptors = kd_utils.get_keypoints_and_descripotrs(
                self.feature_detector, self.descriptor_extractor,
                self.img1.img_gray)
        elif not len(self.img1.key_points) == len(
                self.img1.descriptors
        ):    # not the same, aka, keypoints have been updated, but not descriptors
            print(
                "We are going to use the existing keypoints to calculate the descriptors"
            )
            __, self.img1.descriptors = kd_utils.get_keypoints_and_descripotrs_with_known_keypoints( # note, we will not replace the keypoints at all!!
                self.descriptor_extractor, self.img1.key_points,
                self.img1.img_gray)

        self.img2.key_points, self.img2.descriptors = kd_utils.get_keypoints_and_descripotrs(
            self.feature_detector, self.descriptor_extractor,
            self.img2.img_gray)

    def _get_matches(self):
        self.matches = kd_utils.match_with_type(
            self.matcher, self.img1.descriptors, self.img2.descriptors,
            self.normType)

    def _find_fundamental_matrix(self):
        """Estimates fundamental matrix """
        self.F, self.matches_F, _ = pe_utils.find_F_and_matches(
            self.img1.key_points, self.img2.key_points, self.matches)

    def _refine_with_fundamental_matrix(self):
        """Estimates fundamental matrix """
        self.F, self.matches_F, _ = pe_utils.find_F_and_matches(
            self.img1.key_points, self.img2.key_points, self.matches)
        self.matches = self.matches_F

    def _find_essential_matrix(self):
        """Estimates essential matrix based on fundamental matrix """

        if imutils.is_cv2():
            # note, in OpenCV 2.x we use F to calc E, however, to make the API alike, we do the calculation inside
            self.E, self.matches_E, self.matches_E_bad = pe_utils.find_E_and_matches_cv2(
                self.img1.key_points, self.img2.key_points, self.matches,
                self.K)
            print("E from FundamentalMat:{}".format(self.E))

        elif imutils.is_cv3():
            self.E, self.matches_E, self.matches_E_bad = pe_utils.find_E_and_matches_cv3(
                self.img1.key_points, self.img2.key_points, self.matches,
                self.K)
            print("E from findEssentialMat:{}".format(self.E))

    def _find_camera_matrices_rt(self):

        if imutils.is_cv3():

            print("USE recoverPose in cv3:")

            self.R, self.t, matches_rp_cv3, matches_rp_bad_cv3 = pe_utils.recoverPose_from_E_cv3(
                self.E, self.img1.key_points, self.img2.key_points,
                self.matches_E, self.K)

            pe_utils.DEBUG_Rt(self.R, self.t, "R t from recoverPose")

        elif imutils.is_cv2():
            self.R, self.t, matches_rp, matches_rp_bad = pe_utils.recoverPose_from_E_cv2(
                self.E, self.img1.key_points, self.img2.key_points,
                self.matches_E, self.K)

            pe_utils.DEBUG_Rt(self.R, self.t, "R t from linear algebra")

    def _refine_rt_with_ransac(self, split_num=100, thres=0.5):

        from ransac_Rt import split_matches_and_remove_less_confidence, get_zyxs_ts, get_nice_and_constant_zyxs_ts_list, print_out

        Rs, ts, confidences = split_matches_and_remove_less_confidence(
            self.img1.key_points, self.img2.key_points, self.matches, self.K,
            split_num, thres)

        zyxs_ts = get_zyxs_ts(Rs, ts)

        if len(zyxs_ts) >= 2:    # just one case
            zyxs_ts_refine_list = get_nice_and_constant_zyxs_ts_list(zyxs_ts)
        else:
            zyxs_ts_refine_list = zyxs_ts

        print_out(
            zyxs_ts,
            confidences,
            zyxs_ts_refine_list,
            im1_file_name=self.img1.base_name,
            im2_file_name=self.img2.base_name,
            folder_name=self.output_folder)

        ##############################
        # We should recover R,t here #
        ##############################
        import ransac_Rt
        import Rt_transform
        mean_values = ransac_Rt.mean_zyxs_ts(zyxs_ts_refine_list)
        self.R = Rt_transform.EulerZYXDegree2R(mean_values[:3].reshape(3, 1))
        self.t = mean_values[3:].reshape(3, 1)

        # E_backward = pe_utils.find_E_from_R_t(self.R, self.t)
        # import refine
        # refine.correctMatches_with_E(self.E, self.K, self.img1.key_points, self.img2.key_points, self.matches)

        # pe_utils.DEBUG_Rt(self.R, self.t, "R t before recoverPose with backward E")
        # self.R, self.t, matches_rp_cv3, matches_rp_bad_cv3 = pe_utils.recoverPose_from_E_cv3(
        #     E_backward, self.img1.key_points, self.img2.key_points,
        #     self.matches, self.K)
        # self.matches = matches_rp_cv3
        # pe_utils.DEBUG_Rt(self.R, self.t, "R t after recoverPose with backward E")

    def _plot_point_cloud(self):

        print("In _plot_point_cloud:")

        Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))

        # self.t = self.t / (self.t[0] / -40.0) # NOTE HERE

        self.R = np.eye(3)
        self.t = np.array([-40, 0, 0]).reshape(3, 1)

        Rt2 = np.hstack([self.R, self.t])

        # t = np.array([-40, 0, -15]).reshape(3, 1)
        # Rt2 = np.hstack((np.eye(3), t))
        # Rt2 = np.hstack((self.R, t))

        self.pts3D_Nx3 = depth_estimation.get_and_plot_point_cloud(
            self.img1.key_points, self.img2.key_points, self.matches,
            self.K_inv, Rt1, Rt2)

    def _get_BIG_DICT_points_pixel_to_3d(self):
        self.remain_kps1, self.remain_kps2 = pe_utils.get_matched_key_points(
            self.img1.key_points, self.img2.key_points, self.matches)

        assert len(self.pts3D_Nx3) == len(self.remain_kps1)

        self.img1.key_points = self.remain_kps1    # we replace the key points to points with corresponding 3d

        self.BIG_DICT_points_pixel_to_3d = dict()
        for i in range(len(self.img1.key_points)):
            self.BIG_DICT_points_pixel_to_3d[self.img1.key_points[
                i]] = self.pts3D_Nx3[i]

        assert len(self.pts3D_Nx3) == len(self.BIG_DICT_points_pixel_to_3d)
        print("self.BIG_DICT_points_pixel_to_3d[self.img1.key_points[0]]".
              format(self.BIG_DICT_points_pixel_to_3d[self.img1.key_points[0]]))

    def _get_pts3D_Nx3_from_BIG_DICT(self):

        self.remain_kps1, self.remain_kps2 = pe_utils.get_matched_key_points(
            self.img1.key_points, self.img2.key_points, self.matches)

        pts3D_Nx3_list = []

        for kp in self.remain_kps1:
            pts3D_Nx3_list.append(self.BIG_DICT_points_pixel_to_3d[kp])

        assert len(pts3D_Nx3_list) == len(self.remain_kps1)

        self.pts3D_Nx3 = np.array(pts3D_Nx3_list)

        assert (self.pts3D_Nx3.shape[1] == 3)

    def _PNPSolver_img2_pts_and_3DPoints(self):

        pts1, pts2, _ = pe_utils.key_points_to_matched_pixel_points(
            self.img1.key_points, self.img2.key_points, self.matches)

        print("pts3D_Nx3.shape:{}, pts1.shape:{}".format(
            self.pts3D_Nx3.shape, pts1.shape))
        assert len(self.pts3D_Nx3) == len(pts1)
        """solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, flags]]]]) -> retval, rvec, tvec"""
        _, rvec, tvec = cv2.solvePnP(self.pts3D_Nx3, pts2, self.K, None)

        # imagePoints = np.ascontiguousarray(pts2[:, :2]).reshape((-1, 1, 2)) # no need, [ref, note in solvePnP](http://www.docs.opencv.org/3.3.0/d9/d0c/group__calib3d.html)
        # _, rvec, tvec = cv2.solvePnP(self.pts3D_Nx3, imagePoints, self.K, None)

        print("rvec:\n{}".format(rvec))
        print("tvec:\n{}".format(tvec))

        R, _ = cv2.Rodrigues(rvec)
        pe_utils.DEBUG_Rt(R, tvec, "R t in _PNPSolver_img2_pts_and_3DPoints")

        from ransac_Rt import get_zyxs_ts, get_nice_and_constant_zyxs_ts_list, print_out

        Rs = [R]
        ts = [tvec]
        confidences = np.array([1])

        zyxs_ts = get_zyxs_ts(Rs, ts)
        # zyxs_ts_refine_list = get_nice_and_constant_zyxs_ts_list(zyxs_ts)
        zyxs_ts_refine_list = zyxs_ts

        print_out(
            zyxs_ts,
            confidences,
            zyxs_ts_refine_list,
            im1_file_name=self.img1.base_name,
            im2_file_name=self.img2.base_name + 'PnP',
            folder_name=self.output_folder)

    def _PNPSolver_with_already_3DPoints(self):
        pass
