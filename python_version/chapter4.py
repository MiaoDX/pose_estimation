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


def main():
    pass
    #scene.plot_optic_flow()
    #scene.draw_epipolar_lines()
    #scene.plot_rectified_images(feat_mode="orb")

    # draw 3D point cloud of fountain
    # use "pan axes" button in pyplot to inspect the cloud (rotate and zoom
    # to convince you of the result)
    # scene.plot_point_cloud(feat_mode="orb")
    #scene.plot_point_cloud()


def test_camera_relocation():

    K = np.array([[8607.8639, 0, 2880.72115], [0, 8605.4303, 1913.87935],
                  [0, 0, 1]])    # Canon5DMarkIII-EF50mm
    d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)

    base_dir = "H:/projects/SLAM/python_code/dataset/our/trajs2/"


    #cameraRelocation = CameraRelocation(K, d, feature_name="BRIEF")
    cameraRelocation = CameraRelocation(K, d, feature_name="ORB",output_folder="20170827")

    cameraRelocation.set_feature_detector_descriptor_extractor("ORB", None, dict(nfeatures=2000))
    #cameraRelocation.set_matcher(True) # this already set with the set fd and de



    strs = [str(x) for x in range(1, 10)]
    strs.extend(['1a', '1b', '1c', '4a', '7a', '7b'])
    #strs = sorted(strs)

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
                print("Somthing went wrong when calc {} and {}".format(im1_file, im2_file))
                continue


if __name__ == '__main__':
    #main()
    test_camera_relocation()
