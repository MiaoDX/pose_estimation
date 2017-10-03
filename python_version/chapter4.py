#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OpenCV with Python Blueprints
    Chapter 4: 3D Scene Reconstruction Using Structure From Motion

    An app to detect and extract structure from motion on a pair of images
    using stereo vision. We will assume that the two images have been taken
    with the same camera, of which we know the internal camera parameters. If
    these parameters are not known, use calibrate.py to estimate them.

    The result is a point cloud that shows the 3D real-world coordinates
    of points in the scene.
"""

import numpy as np

from scene3D import CameraRelocation


def test_camera_relocation_tweak_manaually():
    import traceback

    K = np.array([[8607.8639, 0, 2880.72115], [0, 8605.4303, 1913.87935],
                  [0, 0, 1]])    # Canon5DMarkIII-EF50mm
    d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)

    base_dir = "H:/projects/SLAM/dataset/cartoon_2/"
    out_folder = "20170908_cartoon_2"

    # cameraRelocation = CameraRelocation(K, d, feature_name="BRIEF")

    cameraRelocation = CameraRelocation(
        K, d, feature_name="ORB", output_folder=out_folder)
    cameraRelocation.set_feature_detector_descriptor_extractor(
        "ORB", feature_detector_params=dict(nfeatures=2000))

    strs = [str(x) for x in range(1, 10)]
    strs.extend(['1a', '1b', '1c', '1d', '4a', '4b', '7a', '7b'])
    strs = sorted(strs)

    # strs = ['1', '4', '2', '3', '4', '5', '6', '7', '8', '9']
    strs = ['1', '4', '2']

    i = 0

    im1_file = base_dir + strs[i] + ".jpg"
    print("Using {} as the reference image".format(im1_file))

    cameraRelocation.load_image_left(im1_file)
    for j in range(i + 1, len(strs)):
        im2_file = base_dir + strs[j] + ".jpg"
        print("Using {} as the testing image".format(im2_file))
        try:
            cameraRelocation.forward(im2_file)
        except:
            print("Somthing went wrong when calc {} and {}".format(
                im1_file, im2_file))
            traceback.print_exc()
            continue


def test_only_two_images():
    import traceback

    K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]])    # unrealcv
    d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)

    base_dir = "H:/projects/graduation_project_codebase/ACR/dataset/nine_scene/"
    out_folder = "20170930_test"

    # cameraRelocation = CameraRelocation(K, d, feature_name="BRIEF")

    cameraRelocation = CameraRelocation(
        K, d, feature_name="ORB", output_folder=out_folder)
    cameraRelocation.set_feature_detector_descriptor_extractor(
        "ORB", feature_detector_params=dict(nfeatures=2000))


    strs = ['1_000', '1_000']

    im1_file = base_dir + strs[0] + ".png"
    print("Using {} as the reference image".format(im1_file))

    cameraRelocation.load_image_left(im1_file)

    im2_file = base_dir + strs[1] + ".png"
    try:
        print("Using {} as the testing image".format(im2_file))
        cameraRelocation.forward(im2_file)
    except:
        print("Somthing went wrong when calc {} and {}".format(
            im1_file, im2_file))
        traceback.print_exc()

if __name__ == '__main__':

    #test_camera_relocation_tweak_manaually()
    test_only_two_images()
