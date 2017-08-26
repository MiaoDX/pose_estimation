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

    def __init__(self, K, dist, feature_name="ORB"):
        """Constructor

            This method initializes the scene reconstruction algorithm.

            :param K: 3x3 intrinsic camera matrix
            :param dist: vector of distortion coefficients
        """
        self.K = K
        self.K_inv = np.linalg.inv(K)  # store inverse for fast access
        self.d = dist

        self.img1 = Image()
        self.img2 = Image()

        # Ready to go
        self.set_feature_detector_descriptor_extractor(feature_name)
        self.set_matcher(withFlann=False)


    def forward(self, new_frame_name):

        if self.img1.img_name is None:
            print("Load the left image")
            self._load_image_left(new_frame_name)
            return
        else:
            print("Load the right image")
            self._load_image_right(new_frame_name)

        self._get_keypoints_and_descripotrs()
        self._get_matches()
        self._find_fundamental_matrix()
        self._find_essential_matrix()
        self._find_camera_matrices_rt()


    def _load_image_left(self, img_path):
        img_color, img_gray = self._load_image(img_path)
        self.img1 = Image(img_path, img_color, img_gray)

    def _load_image_right(self, img_path):
        img_color, img_gray = self._load_image(img_path)
        self.img2 = Image(img_path, img_color, img_gray)

    def _load_image(self, img_path, use_pyr_down=False, target_width = 600):
        img_color = cv2.imread(img_path)
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # make sure images are valid
        if img is None:
            sys.exit("Image " + img_path + " could not be loaded.")

        # scale down image if necessary to something close to target_width wide (px)
        if use_pyr_down and img.shape[1] > target_width:
            while img.shape[1] > 2*target_width:
                img = cv2.pyrDown(img)
        # undistort the images
        img = cv2.undistort(img, self.K, self.d)
        return img_color, img


    def set_feature_detector_descriptor_extractor(self, featurename="ORB", descriptor_extractor_name=None, feature_detector_params=None, descriptor_extractor_params=None):
        self.feature_detector, self.descriptor_extractor, self.normType = kd_utils.get_feature_detector_descriptor_extractor(featurename, descriptor_extractor_name, feature_detector_params, descriptor_extractor_params)

    def set_matcher(self, withFlann=False):
        self.matcher = kd_utils.get_matcher(self.normType, withFlann)

    def _get_keypoints_and_descripotrs(self):

        if self.img1.key_points is None:
            self.img1.key_points, self.img1.descriptors = kd_utils.get_keypoints_and_descripotrs(self.feature_detector,
                                                                                             self.descriptor_extractor,
                                                                                             self.img1.img_gray)

        self.img2.key_points, self.img2.descriptors = kd_utils.get_keypoints_and_descripotrs(self.feature_detector,
                                                                                             self.descriptor_extractor,
                                                                                             self.img2.img_gray)

    def _get_matches(self):
        self.matches = kd_utils.match_with_type(self.matcher, self.img1.descriptors, self.img2.descriptors, self.normType)


    def _find_fundamental_matrix(self):
        """Estimates fundamental matrix """
        self.F, self.matches_F, _ = pe_utils.find_F_and_matches(self.img1.key_points, self.img2.key_points, self.matches)

    def _refine_with_fundamental_matrix(self):
        """Estimates fundamental matrix """
        self.F, self.matches_F, _ = pe_utils.find_F_and_matches(self.img1.key_points, self.img2.key_points, self.matches)
        self.matches = self.matches_F

    def _find_essential_matrix(self):
        """Estimates essential matrix based on fundamental matrix """

        if imutils.is_cv2():
            # note, in OpenCV 2.x we use F to calc E, however, to make the API alike, we do the calculation inside
            self.E, self.matches_E, self.matches_E_bad = pe_utils.find_E_and_matches_cv2(self.img1.key_points, self.img2.key_points,
                                                                                         self.matches,
                                                                                         self.K)
            print("E from FundamentalMat:{}".format(self.E))

        elif imutils.is_cv3():
            self.E, self.matches_E, self.matches_E_bad = pe_utils.find_E_and_matches_cv3(self.img1.key_points, self.img2.key_points,
                                                                                         self.matches,
                                                                                         self.K)
            print("E from findEssentialMat:{}".format(self.E))


    def _find_camera_matrices_rt(self):

        if imutils.is_cv3():

            print("USE recoverPose in cv3:")

            self.R, self.t, matches_rp_cv3, matches_rp_bad_cv3 = pe_utils.recoverPose_from_E_cv3(self.E, self.img1.key_points,
                                                                                                 self.img2.key_points, self.matches_E,
                                                                               self.K)


            pe_utils.DEBUG_Rt(self.R, self.t, "R t from recoverPose")

        elif imutils.is_cv2():
            self.R, self.t, matches_rp, matches_rp_bad = pe_utils.recoverPose_from_E_cv2(self.E, self.img1.key_points,
                                                                                                 self.img2.key_points, self.matches_E,
                                                                               self.K)

            pe_utils.DEBUG_Rt(self.R, self.t, "R t from linear algebra")




