/*
 * Change from [gaoxiang's pose_estimation_2d2d](https://github.com/gaoxiang12/slambook/blob/master/ch7/pose_estimation_2d2d.cpp)
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <chrono>
#include <numeric>      // std::iota
#include <random>       // std::default_random_engine
#include <map> 
#include <math.h>

using namespace std;
using namespace cv;

void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

void find_feature_matches_from_keypoints (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

void DebugMatchedKeyPoints (
    const Mat& img_1, const Mat& img_2,
    const std::vector<KeyPoint>& keypoints_1,
    const std::vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches
);


// [Copy from SFM](https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/fundamental.cpp)
void essentialFromFundamental ( const Mat &F,
    const Mat &K1,
    const Mat &K2,
    Mat& E );


// 像素坐标转相机归一化坐标
Point2f pixel2cam ( const Point2f& p, const Mat& K );

void kp2pts ( const std::vector<KeyPoint>& keypoints_1,
    const std::vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches,
    vector<Point2f>& points1,
    vector<Point2f>& points2
);