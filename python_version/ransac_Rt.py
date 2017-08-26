#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Usage: python ransac_Rt.py -l 1.jpg -r 2.jpg

When we have many many points pairs, the findEssentialMat and recoverPose will give us only one answer anyway (even it do involve RANSAC inside),
which will definitely loose the benefit of numbers and we can do some `RANSAC` or `sliding-window` approaches outside by splitting up our points.

Let's say, we have 1000 pairs of matched points, there can be ways to do it:

===========
Dead sample:
1. Calculate every 100 points and get 10 answers
2. Remove the answers with less confidence, aka, the less percentage of points passing the cheirality check (in recoverPose)
3. Get the MEAN of the answers
===========
With clustering:
1. Calculate every 100 points and get 10 answers
2. Remove the answers with less confidence
3. Get the main centroid of the remaining answers
===========
Pruning by reprojection:
1. Calculate every 100 points and get 10 answers
2. For each answer, calc the reprojection error (or the epipolar equation) on all points and filter out some answers
3. Get the main centroid of the remaining answers
===========

"""

import cv2
import numpy as np
import pose_estimation_utils as pe_utils


def split_the_matches(matches, step):
    for i in range(len(matches) // step + 1):
        yield matches[i * step:(i + 1) * step]


def split_matches_and_remove_less_confidence(kps1, kps2, matches, K, splitnum, conf_thresh=0.7):
    Rs = []
    ts = []
    confidences = []

    splited_matches = split_the_matches(matches, splitnum)

    for chosen_matches in splited_matches:
        if len(chosen_matches) < 5:
            print("Less than 5 points, just return")
            return

        E, matches_E, _ = pe_utils.find_E_and_matches_cv3(kps1, kps2, chosen_matches, K)

        R, t, matches_rp, matches_rp_bad = pe_utils.recoverPose_from_E_cv3(E, kps1, kps2, matches_E, K)

        conf = len(matches_rp) / len(chosen_matches)
        if conf >= conf_thresh:
            Rs.append(R)
            ts.append(t)
            confidences.append(conf)

    return np.array(Rs), np.array(ts), np.array(confidences)


def get_zyxs_ts(Rs, ts):
    zyxs = pe_utils.Rs2zyxs(Rs)
    zyxs_ts = np.hstack([zyxs, ts]).reshape(len(zyxs), 6)

    return zyxs_ts


def mean_zyxs_ts(zyxs_ts):
    return zyxs_ts.mean(axis=0)


def AgglomerativeClustering_linkage_average(X, cluster_num):
    """
    :ref http://scikit-learn.org/stable/auto_examples/cluster/plot_digits_linkage.html#sphx-glr-auto-examples-cluster-plot-digits-linkage-py
    :param X:
    :param cluster_num:
    :return:
    """
    from time import time
    from sklearn.cluster import AgglomerativeClustering

    print("cluster_num:", cluster_num)
    clustering = AgglomerativeClustering(linkage='average', n_clusters=cluster_num)
    t0 = time()
    clustering.fit(X)
    print("Using linkage {}, time cost {}".format('average', time() - t0))

    return list(clustering.labels_)


def get_intersection_label_index(new_labels, last_chosen_index_list):
    from scipy.stats import mode
    lable_mode = mode(new_labels).mode
    mode_indexes = np.where(new_labels == lable_mode)
    chosen_list = np.intersect1d(mode_indexes, last_chosen_index_list)

    return chosen_list


def get_nice_and_constant_zyxs_ts_list(zyxs_ts, accept_mode_ration=0.6):
    # assert zyxs_ts.shape

    all_len = len(zyxs_ts)
    last_chosen_index_list = list(range(all_len))
    for i in range(2, 6):
        new_labels = AgglomerativeClustering_linkage_average(zyxs_ts, i)
        chosen_index_list = get_intersection_label_index(new_labels, last_chosen_index_list)

        if len(chosen_index_list) / all_len < accept_mode_ration:
            break
        else:
            last_chosen_index_list = chosen_index_list

    # return last_chosen_index_list
    zyxs_ts_refine = zyxs_ts[last_chosen_index_list]

    return zyxs_ts_refine


"""
=======================
========TESTING========
=======================
"""

import keypoints_descriptors_utils as kd_utils


def test_remove_less_confidence(kps1, kps2, matches, K, split_num=50, thres=0.7):
    start_lsd = cv2.getTickCount()
    freq = cv2.getTickFrequency()

    test_num = 10
    for i in range(test_num):
        Rs, ts, confidences = split_matches_and_remove_less_confidence(kps1, kps2, matches, K, split_num, thres)

    duration_ms_lsd = (cv2.getTickCount() - start_lsd) * 1000 / freq / test_num
    print("Elapsed time for split_matches_and_remove_less_confidence: {} ms".format(duration_ms_lsd))

    zyxs_ts = get_zyxs_ts(Rs, ts)

    print("The answers num:{} of total:{}".format(len(Rs), len(matches) // split_num))
    for R, t, con in zip(Rs, ts, confidences):
        print("=============")
        print("Confidence:{}".format(con))
        pe_utils.DEBUG_Rt(R, t, con)
        print("=============")

    zyxs_ts_mean = mean_zyxs_ts(zyxs_ts)
    print("rs_ts_mean:{}".format(zyxs_ts_mean))

    return Rs, ts, confidences, zyxs_ts_mean


def print_out(zyxs_ts, confidences, zyxs_ts_refine_list, im1_file_name="im1", im2_file_name="im2"):
    import sys
    savedStdout = sys.stdout  # 保存标准输出流

    with open(im1_file_name + im2_file_name + '.txt', 'w') as f:
        sys.stdout = f  # 标准输出重定向至文件

        np.set_printoptions(precision=6)
        print(im1_file_name + im2_file_name)
        print("zyxs_ts_and_confidences:\n")

        zyxs_ts_confs = np.concatenate((zyxs_ts, confidences.reshape(len(zyxs_ts), 1)), axis=1)

        print(zyxs_ts_confs)

        print("mean_zyxs_ts:{}".format(mean_zyxs_ts(zyxs_ts)))
        print(zyxs_ts_refine_list)
        print("zyxs_ts_refine_mean:{}".format(mean_zyxs_ts(zyxs_ts_refine_list)))

    sys.stdout = savedStdout  # 恢复标准输出流


def process(im1_file, im2_file, im1_file_name_short, im2_file_name_short, K):
    im1 = cv2.imread(im1_file, 0)
    im2 = cv2.imread(im2_file, 0)

    feature_detector, descriptor_extractor, _ = kd_utils.get_feature_detector_descriptor_extractor("ORB",
                                                                                                   feature_detector_params=dict(
                                                                                                       nfeatures=2000))
    kps1, des1 = kd_utils.get_keypoints_and_descripotrs(feature_detector, descriptor_extractor, im1)
    kps2, des2 = kd_utils.get_keypoints_and_descripotrs(feature_detector, descriptor_extractor, im2)

    matcher = kd_utils.get_matcher(cv2.NORM_HAMMING)
    all_matches = kd_utils.match_with_type(matcher, des1, des2, normType=cv2.NORM_HAMMING)

    # test_remove_less_confidence_time()
    Rs, ts, confidences, zyxs_ts_mean = test_remove_less_confidence(kps1, kps2, all_matches, K, split_num=100,
                                                                    thres=0.7)

    zyxs_ts = get_zyxs_ts(Rs, ts)
    zyxs_ts_refine_list = get_nice_and_constant_zyxs_ts_list(zyxs_ts)

    print_out(zyxs_ts, confidences, zyxs_ts_refine_list, im1_file_name=im1_file_name_short,
              im2_file_name=im2_file_name_short)


if __name__ == "__main__":

    import argparse

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--image1", required=True,
                    help="path to input image1")
    ap.add_argument("-r", "--image2", required=True,
                    help="path to input image2")
    args = vars(ap.parse_args())

    base_dir = "H:/projects/SLAM/python_code/dataset/our/trajs2/"

    im1_file = base_dir + args["image1"]
    im2_file = base_dir + args["image2"]

    # print("{}\n{}".format(im1_file, im2_file))
    print("{}\n{}".format(args["image1"], args["image2"]))

    K = np.array([[8607.8639, 0, 2880.72115], [0, 8605.4303, 1913.87935], [0, 0, 1]])  # Canon5DMarkIII-EF50mm

    # process(im1_file, im2_file, K)

    strs = [str(x) for x in range(1, 10)]
    strs.extend(['1a', '1b', '1c', '4a', '7a', '7b'])

    for i in range(len(strs) - 1):
        for j in range(i + 1, len(strs)):

            try:

                im1_file = base_dir + strs[i] + ".jpg"
                im2_file = base_dir + strs[j] + ".jpg"
                process(im1_file, im2_file, strs[i] + '-', strs[j], K)
                # input()
            except:
                continue
