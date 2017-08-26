#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The helper function of keypoints and descriptors
"""

import cv2
import imutils


def get_feature_detector_descriptor_extractor(feature_detector_name=str(),
                                              descriptor_extractor_name=None,
                                              feature_detector_params=None,
                                              descriptor_extractor_params=None):
    """
    :param feature_detector_name:
    :param descriptor_extractor_name:
    :param feature_detector_params: dict(nfeatures=1000) for ORB
    :param descriptor_extractor_params:
    :return:
    """
    assert len(feature_detector_name) != 0
    if feature_detector_params == None:
        feature_detector_params = dict()
    if descriptor_extractor_params == None:
        descriptor_extractor_params = dict()

    feature_detector_name = feature_detector_name.upper()

    normType = cv2.NORM_L2

    if feature_detector_name == "ORB" or feature_detector_name == "BRIEF" or feature_detector_name == "BRISK":
        normType = cv2.NORM_HAMMING

    feature_detector = descriptor_extractor = None
    if feature_detector_name == "ORB":
        assert descriptor_extractor_name is None and len(
            descriptor_extractor_params) == 0
        if imutils.is_cv2():
            feature_detector = descriptor_extractor = cv2.ORB(
                **feature_detector_params)
        else:
            feature_detector = descriptor_extractor = cv2.ORB_create(
                **feature_detector_params)

    elif feature_detector_name == "BRIEF":
        assert descriptor_extractor_name is None and len(
            descriptor_extractor_params) == 0
        if imutils.is_cv2():
            feature_detector = cv2.StarDetector(**feature_detector_params)
            #descriptor_extractor = cv2.BriefDescriptorExtractor(**descriptor_extractor_params)
            descriptor_extractor = cv2.DescriptorExtractor_create("BRIEF")
        else:
            feature_detector = cv2.xfeatures2d.StarDetector_create(
                **feature_detector_params)
            descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                **descriptor_extractor_params)

    elif feature_detector_name == "BRISK":
        assert descriptor_extractor_name is None and len(
            descriptor_extractor_params) == 0
        if imutils.is_cv2():
            feature_detector = descriptor_extractor = cv2.BRISK(
                **feature_detector_params)
        else:
            feature_detector = descriptor_extractor = cv2.BRISK_create(
                **feature_detector_params)

    elif feature_detector_name == "SURF":
        assert descriptor_extractor_name is None and len(
            descriptor_extractor_params) == 0
        if imutils.is_cv2():
            feature_detector = descriptor_extractor = cv2.SURF(
                **feature_detector_params)
        else:
            feature_detector = descriptor_extractor = cv2.xfeatures2d.SURF_create(
                **feature_detector_params)

    elif feature_detector_params == "SIFT":
        assert descriptor_extractor_name is None and len(
            descriptor_extractor_params) == 0
        if imutils.is_cv2():
            feature_detector = descriptor_extractor = cv2.SIFT(
                **feature_detector_params)
        else:
            feature_detector = descriptor_extractor = cv2.xfeatures2d.SIFT_create(
                **feature_detector_params)

    else:
        print(
            "Seems we have not predefined the target feature_detector and descriptor_extractor"
        )

    return feature_detector, descriptor_extractor, normType


def get_matcher(normType=cv2.NORM_L2, withFlann=False):
    """
    http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    :param normType:
    :param withFlann:
    :return:
    """
    if not withFlann:
        # create BFMatcher object
        return cv2.BFMatcher(normType, crossCheck=True)

    # with flann
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_LSH = 6
    FLANN_INDEX_AUTOTUNED = 255
    search_params = dict(checks=50)    # or pass empty dictionary

    if normType == cv2.NORM_L2:
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        return cv2.FlannBasedMatcher(index_params, search_params)

    # normType == NORM_HAMMING

    index_params_orb = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,    # 12
        key_size=12,    # 20
        multi_probe_level=1)    # 2

    index_params_orb2 = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=12,    # 12
        key_size=20,    # 20
        multi_probe_level=2)    # 2
    """
    # FAILED
    index_params_auto = dict(algorithm=FLANN_INDEX_AUTOTUNED,
                             target_precision=0.9,
                             build_weight=0.01,
                             memory_weight=0,
                             sample_fraction=0.1)
    # flann = cv2.FlannBasedMatcher(index_params_auto, search_params)
    """

    return cv2.FlannBasedMatcher(index_params_orb, search_params)
    #cv2.FlannBasedMatcher(index_params_orb2, search_params)


def get_keypoints_and_descripotrs(feature_detector, descriptor_extractor, img):
    #kps = feature_detector.detect(img, None)
    kps = feature_detector.detect(img)
    # NOTE: it will output another key-points, but the are presenting the same thing
    keypoints, descriptors = descriptor_extractor.compute(img, kps)
    return keypoints, descriptors


def match_with_type(matcher, des1, des2, normType=cv2.NORM_L2):

    if normType == cv2.NORM_HAMMING:
        # Match descriptors.
        matches = matcher.match(des1, des2)

        # Sort them in the order of their distance.
        matches_sorted = sorted(matches, key=lambda x: x.distance)
        print("Max distance:{}, min distance:{}".format(matches_sorted[
            -1].distance, matches_sorted[0].distance))
        # 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
        prune_dis = max(matches_sorted[0].distance, 30.0)

        matches_good = list(filter(lambda x: x.distance <= prune_dis, matches))

        print(
            "Matches with prune:{}->{}".format(len(matches), len(matches_good)))

    else:
        # NORM_L2
        matches = matcher.knnMatch(des1, des2, k=2)
        # Need to draw only good matches, so create a mask
        matches_good = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matches_good.append(m)

        print("Matches with ratio test:{}->{}".format(
            len(matches), len(matches_good)))

    return matches_good


def DEBUG_feature_detector(detector):
    """
    https://stackoverflow.com/questions/13658185/setting-orb-parameters-in-opencv-with-python
    :param detector:
    :return:
    """
    if imutils.is_cv2():

        params = detector.getParams()
        print("Detector parameters (dict):", params)
        for param in params:
            ptype = detector.paramType(param)
            if ptype == 0:
                print("{} = {}".format(param, detector.getInt(param)))
            elif ptype == 2:
                print("{} = {}".format(param, detector.getDouble(param)))
        return

    # for OpenCV 3.x
    attributes = dir(detector)
    print("Detector parameters (dict):", attributes)
    for attribute in attributes:
        if not attribute.startswith("get"):
            continue
        param = attribute.replace("get", "")
        get_param = getattr(detector, attribute)
        val = get_param()
        print(param, '=', val)
