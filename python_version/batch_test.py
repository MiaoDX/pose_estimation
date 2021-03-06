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


def test_all():
    dir1 = "H:/projects/SLAM/dataset/cartoon_1/"
    dir2 = "H:/projects/SLAM/dataset/cartoon_2/"
    dir3 = "H:/projects/SLAM/dataset/Marx_1/"
    dir4 = "H:/projects/SLAM/dataset/Marx_2/"
    base_dirs = [dir3, dir4]

    # base_dirs = [dir3]

    for dir in base_dirs:
        test_camera_relocation_5_points_ransac(dir)


def test_camera_relocation_5_points_ransac(base_dir, log_dir="20170905_2_"):
    import traceback

    K = np.array([[8607.8639, 0, 2880.72115], [0, 8605.4303, 1913.87935],
                  [0, 0, 1]])    # Canon5DMarkIII-EF50mm
    d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)

    # cameraRelocation = CameraRelocation(K, d, feature_name="BRIEF")

    # dir = "20170905_2_" + base_dir.split('/')[-2]
    dir = log_dir + base_dir.split('/')[-2]

    cameraRelocation = CameraRelocation(
        K, d, feature_name="ORB", output_folder=dir)
    cameraRelocation.set_feature_detector_descriptor_extractor(
        "ORB", feature_detector_params=dict(nfeatures=2000))

    strs = [str(x) for x in range(1, 10)]
    strs.extend(['1a', '1b', '1c', '1d', '4a', '4b', '7a', '7b'])
    strs = sorted(strs)

    for i in range(len(strs) - 1):
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


if __name__ == '__main__':

    base_dir = "H:/projects/SLAM/dataset/cartoon_2/"
    test_camera_relocation_5_points_ransac(
        base_dir, log_dir="20170908_test_mask_E_")

    # test_all()
