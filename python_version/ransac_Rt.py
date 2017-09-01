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
    """

    :param matches:
    :param step:
    :return:
    >>> l = list(range(12)) # note, the input should be matches list, however, that is not so easy to mock
    >>> sl = split_the_matches(l, 4) # the last will be []
    >>> sl = list(sl)
    >>> len(sl)
    4
    >>> sl[0]
    [0, 1, 2, 3]
    >>> sl_2 = split_the_matches(l, 5)
    >>> sl_2 = list(sl_2)
    >>> len(sl_2)
    3
    >>> sl_2[0]
    [0, 1, 2, 3, 4]
    """
    for i in range(len(matches) // step + 1):
        yield matches[i * step:(i + 1) * step]


def split_matches_and_remove_less_confidence(kps1,
                                             kps2,
                                             matches,
                                             K,
                                             splitnum=100,
                                             conf_thresh=0.7):
    assert kps1 is not None and kps2 is not None and matches is not None and K is not None

    Rs = []
    ts = []
    confidences = []

    splited_matches = split_the_matches(matches, splitnum)

    for chosen_matches in splited_matches:
        if len(chosen_matches) < 5:
            print("Less than 5 points, just return")
            break

        E, matches_E, _ = pe_utils.find_E_and_matches_cv3(
            kps1, kps2, chosen_matches, K)

        R, t, matches_rp, matches_rp_bad = pe_utils.recoverPose_from_E_cv3(
            E, kps1, kps2, matches_E, K)

        conf = len(matches_rp) / len(chosen_matches)
        if conf >= conf_thresh:
            Rs.append(R)
            ts.append(t)
            confidences.append(conf)

    return np.array(Rs), np.array(ts), np.array(confidences)


def get_zyxs_ts(Rs, ts):
    assert Rs is not None and ts is not None
    assert Rs[0].shape == (3, 3)
    assert ts[0].shape == (3, 1)

    zyxs = pe_utils.Rs2zyxs(Rs)
    zyxs_ts = np.hstack([zyxs, ts]).reshape(len(zyxs), 6)

    return zyxs_ts


def mean_zyxs_ts(zyxs_ts):
    """
    :param zyxs_ts:
    :return:
    >>> zyxs_ts = np.array(range(6)).reshape(1,6)
    >>> list(np.int8(mean_zyxs_ts(zyxs_ts))) # use int for easy compare
    [0, 1, 2, 3, 4, 5]
    >>> zyxs_ts = np.array(range(12)).reshape(2,6)
    >>> list(np.int8(mean_zyxs_ts(zyxs_ts))) # use int for easy compare
    [3, 4, 5, 6, 7, 8]
    """
    assert zyxs_ts[0].shape == (6,)
    return zyxs_ts.mean(axis=0)    # mean in vertical


def AgglomerativeClustering_linkage_average(X, cluster_num):
    """
    :ref http://scikit-learn.org/stable/auto_examples/cluster/plot_digits_linkage.html#sphx-glr-auto-examples-cluster-plot-digits-linkage-py
    :param X:
    :param cluster_num:
    :return:
    >>> a1 = [0, 0, 0, 0, 0, 0]
    >>> a2 = [0, 0, 0, 0, 1, 0]
    >>> a3 = [0, 0, 0, 0, 0, 1]
    >>> a4 = [0, 0, 0, 0, -1, 0]
    >>> a5 = [0, 0, 0, 0, 0, -1]  # above are alike

    >>> a6 = [0, 0, 0, 0, 10, 0]
    >>> a7 = [0, 0, 0, 0, 0, 10]
    >>> a8 = [0, 0, 0, 0, -10, 0]
    >>> a9 = [0, 0, 0, 0, 0, -10]

    >>> X = [a1, a2, a3, a4, a5, a5, a6, a7, a8, a9]
    >>> ag1 = AgglomerativeClustering_linkage_average(X, 2) #doctest: +ELLIPSIS
    >>> ag1[:5] == list(np.ones(5)*ag1[0])
    True
    >>> ag1 = AgglomerativeClustering_linkage_average(X, 3) #doctest: +ELLIPSIS
    >>> ag1[:5] == list(np.ones(5)*ag1[0])
    True
    >>> ag1 = AgglomerativeClustering_linkage_average(X, 4) #doctest: +ELLIPSIS
    >>> ag1[:5] == list(np.ones(5)*ag1[0])
    True
    >>> ag1 = AgglomerativeClustering_linkage_average(X, 5) #doctest: +ELLIPSIS
    >>> ag1[:5] == list(np.ones(5)*ag1[0])
    True
    """
    from time import time
    from sklearn.cluster import AgglomerativeClustering

    print("cluster_num:", cluster_num)
    clustering = AgglomerativeClustering(
        linkage='average', n_clusters=cluster_num)
    t0 = time()
    clustering.fit(X)
    print("Using linkage {}, time cost {}".format('average', time() - t0))

    return list(clustering.labels_)


def AgglomerativeClustering_linkage_average_with_xalglib(X, cluster_num):
    """
    [clst_linkage example](http://www.alglib.net/translator/man/manual.cpython.html)
    :param X:
    :param cluster_num:
    :return:
    >>> a1 = [0, 0, 0, 0, 0, 0]
    >>> a2 = [0, 0, 0, 0, 1, 0]
    >>> a3 = [0, 0, 0, 0, 0, 1]
    >>> a4 = [0, 0, 0, 0, -1, 0]
    >>> a5 = [0, 0, 0, 0, 0, -1]  # above are alike

    >>> a6 = [0, 0, 0, 0, 10, 0]
    >>> a7 = [0, 0, 0, 0, 0, 10]
    >>> a8 = [0, 0, 0, 0, -10, 0]
    >>> a9 = [0, 0, 0, 0, 0, -10]

    >>> X = [a1, a2, a3, a4, a5, a5, a6, a7, a8, a9]
    >>> ag1 = AgglomerativeClustering_linkage_average_with_xalglib(X, 2) #doctest: +ELLIPSIS
    >>> ag1[:5] == list(np.ones(5)*ag1[0])
    True
    >>> ag1 = AgglomerativeClustering_linkage_average_with_xalglib(X, 3) #doctest: +ELLIPSIS
    >>> ag1[:5] == list(np.ones(5)*ag1[0])
    True
    >>> ag1 = AgglomerativeClustering_linkage_average_with_xalglib(X, 4) #doctest: +ELLIPSIS
    >>> ag1[:5] == list(np.ones(5)*ag1[0])
    True
    >>> ag1 = AgglomerativeClustering_linkage_average_with_xalglib(X, 5) #doctest: +ELLIPSIS
    >>> ag1[:5] == list(np.ones(5)*ag1[0])
    True
    """
    from time import time
    import xalglib

    t0 = time()
    s = xalglib.clusterizercreate()
    xalglib.clusterizersetpoints(s, X, 2) # NORM_L2
    xalglib.clusterizersetahcalgo(s, 2) # unweighted average linkage
    rep = xalglib.clusterizerrunahc(s)
    cidx, cz = xalglib.clusterizergetkclusters(rep, cluster_num)
    # print(cidx)


    print("Using linkage {}, time cost {}".format('average', time() - t0))

    return list(cidx)


def get_intersection_sample_index(new_labels, last_chosen_index_list):
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
        chosen_index_list = get_intersection_sample_index(
            new_labels, last_chosen_index_list)

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


def test_remove_less_confidence(kps1,
                                kps2,
                                matches,
                                K,
                                split_num=100,
                                thres=0.7,
                                test_time=10):
    start_lsd = cv2.getTickCount()
    freq = cv2.getTickFrequency()

    for i in range(test_time):
        Rs, ts, confidences = split_matches_and_remove_less_confidence(
            kps1, kps2, matches, K, split_num, thres)

    duration_ms_lsd = (cv2.getTickCount() - start_lsd) * 1000 / freq / test_time
    print("Elapsed time for split_matches_and_remove_less_confidence: {} ms".
          format(duration_ms_lsd))

    return Rs, ts, confidences


def print_out(zyxs_ts,
              confidences,
              zyxs_ts_refine_list,
              im1_file_name="im1",
              im2_file_name="im2",
              folder_name='.'):
    folder_name += '/'
    file_name = folder_name + im1_file_name + '-' + im2_file_name + '.txt'

    import os
    if os.path.isdir(folder_name):
        print(
            "Folder:{} already exists, maybe overwriting!!".format(folder_name))
    else:
        print("Going to create folder: {}".format(folder_name))
        os.mkdir(folder_name)

    if os.path.isfile(file_name):
        print("File already exists, maybe overwriting!!")

    import sys
    savedStdout = sys.stdout    # 保存标准输出流

    with open(file_name, 'w') as f:
        sys.stdout = f    # 标准输出重定向至文件

        np.set_printoptions(precision=5)
        print(im1_file_name + '-' + im2_file_name)
        print("zyxs_ts_and_confidences:\n")

        zyxs_ts_confs = np.concatenate(
            (zyxs_ts, confidences.reshape(len(zyxs_ts), 1)), axis=1)

        print(zyxs_ts_confs)

        print("mean_zyxs_ts:{}".format(mean_zyxs_ts(zyxs_ts)))
        print(zyxs_ts_refine_list)
        print(
            "zyxs_ts_refine_mean:{}".format(mean_zyxs_ts(zyxs_ts_refine_list)))

    sys.stdout = savedStdout    # 恢复标准输出流


def process(im1_file, im2_file, im1_file_name_short, im2_file_name_short, K):

    print("In process, {}-{}".format(im1_file, im2_file))

    im1_color = cv2.imread(im1_file)
    im2_color = cv2.imread(im2_file)
    im1 = cv2.cvtColor(im1_color, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2_color, cv2.COLOR_BGR2GRAY)

    feature_detector, descriptor_extractor, _ = kd_utils.get_feature_detector_descriptor_extractor(
        "ORB", feature_detector_params=dict(nfeatures=2000))

    kps1, des1 = kd_utils.get_keypoints_and_descripotrs(
        feature_detector, descriptor_extractor, im1)
    kps2, des2 = kd_utils.get_keypoints_and_descripotrs(
        feature_detector, descriptor_extractor, im2)

    matcher = kd_utils.get_matcher(cv2.NORM_HAMMING)
    all_matches = kd_utils.match_with_type(
        matcher, des1, des2, normType=cv2.NORM_HAMMING)

    print("kps1:{}, kps2:{}, matches:{}".format(
        len(kps1), len(kps2), len(all_matches)))

    Rs, ts, confidences = test_remove_less_confidence(
        kps1, kps2, all_matches, K, split_num=100, thres=0.7, test_time=1)

    zyxs_ts = get_zyxs_ts(Rs, ts)
    zyxs_ts_refine_list = get_nice_and_constant_zyxs_ts_list(zyxs_ts)

    print_out(
        zyxs_ts,
        confidences,
        zyxs_ts_refine_list,
        im1_file_name=im1_file_name_short,
        im2_file_name=im2_file_name_short)


def run():

    base_dir = "H:/projects/SLAM/python_code/dataset/our/trajs2/"

    K = np.array([[8607.8639, 0, 2880.72115], [0, 8605.4303, 1913.87935],
                  [0, 0, 1]])    # Canon5DMarkIII-EF50mm

    strs = [str(x) for x in range(1, 10)]
    strs.extend(['1a', '1b', '1c', '4a', '7a', '7b'])
    #strs = sorted(strs)

    strs = strs[:2]

    for i in range(len(strs) - 1):
        for j in range(i + 1, len(strs)):
            im1_file = base_dir + strs[i] + ".jpg"
            im2_file = base_dir + strs[j] + ".jpg"
            try:
                process(im1_file, im2_file, strs[i], strs[j], K)
                # input()
            except:
                print("Somthing went wrong when calc {} and {}".format(
                    im1_file, im2_file))
                continue


if __name__ == "__main__":

    import doctest
    old_print = print # the print is not necessary in doctest
    print = lambda *args, **kwargs: None    # we are doing this to suppress the output, so we can do it easy for testing doc
    doctest.testmod(verbose=True)

    print = old_print
    run()
