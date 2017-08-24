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
    # camera matrix and distortion coefficients
    # can be recovered with calibrate.py
    # but the examples used here are already undistorted, taken with a camera
    # of known K

    # K = np.array([[2759.48/4, 0, 1520.69/4, 0, 2764.16/4,
    #                1006.81/4, 0, 0, 1]]).reshape(3, 3)

    K = np.array([[8607.8639, 0, 2880.72115], [0, 8605.4303, 1913.87935], [0, 0, 1]])  # Canon5DMarkIII-EF50mm

    d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)
    #scene = SceneReconstruction3D(K, d)

    # load a pair of images for which to perform SfM
    #scene.load_image_pair("fountain_dense/0004.png", "fountain_dense/0005.png")
    #scene.load_image_pair("H:/projects/SLAM/python_code/dataset/our/trajs2/1.jpg", "H:/projects/SLAM/python_code/dataset/our/trajs2/4.jpg")


    # when these is activated, the plot_point_cloud fail somewhat

    #scene.plot_optic_flow()
    #scene.draw_epipolar_lines()
    #scene.plot_rectified_images(feat_mode="orb")

    # draw 3D point cloud of fountain
    # use "pan axes" button in pyplot to inspect the cloud (rotate and zoom
    # to convince you of the result)
    # scene.plot_point_cloud(feat_mode="orb")
    #scene.plot_point_cloud()


def test_camera_relocation():


    K = np.array([[8607.8639, 0, 2880.72115], [0, 8605.4303, 1913.87935], [0, 0, 1]])  # Canon5DMarkIII-EF50mm
    d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)

    base_dir = "H:/projects/SLAM/python_code/dataset/our/trajs2/"

    im1_file = base_dir + '1.jpg'
    im2_file = base_dir + '2.jpg'
    im3_file = base_dir + '3.jpg'
    im4_file = base_dir + '4.jpg'

    im_files = [im1_file, im2_file, im3_file, im4_file]

    cameraRelocation = CameraRelocation(K, d, feature_name="BRIEF")

    #cameraRelocation.set_feature_detector_descriptor_extractor("ORB", None, dict(nfeatures=1000))
    #cameraRelocation.set_matcher(True) # this should use along with the set fd and de


    #cameraRelocation.forward(im1_file)
    #cameraRelocation.forward(im2_file)

    for im_file in im_files[:2]:
        cameraRelocation.forward(im_file)



if __name__ == '__main__':
    #main()
    test_camera_relocation()