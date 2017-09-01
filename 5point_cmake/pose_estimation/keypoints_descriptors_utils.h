#pragma once

#include "pose_estimation.h"

using namespace std;
using namespace cv;

void getFeatureDetectorDescriptorExtractor (Ptr<FeatureDetector>& fd, Ptr<DescriptorExtractor>& de, const string featurename = "SIFT");

void extractFeaturesAndDescriptors (const Ptr<FeatureDetector>& fd, const Ptr<DescriptorExtractor>& de, const Mat& im1, const Mat& im2,
                                    vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2, Mat& dsp1, Mat& dsp2 );

Ptr<DescriptorMatcher> getMatchTypeNormal ( const int normType = NORM_L2 );

Ptr<DescriptorMatcher> getMatchTypeFlann ( const int normType = NORM_L2 );

void match_with_NORM_HAMMING ( const Ptr<DescriptorMatcher>& matcher, const Mat& des1, const Mat& des2,
                               vector<DMatch>& matches, double threshold_dis = 30.0 );

void match_with_knnMatch ( const Ptr<DescriptorMatcher>& matcher, const Mat& des1, const Mat& des2,
                           vector<DMatch>& matches, float minRatio = 1.f / 1.5f );


void kp2pts ( const std::vector<KeyPoint>& keypoints_1,
              const std::vector<KeyPoint>& keypoints_2,
              const std::vector< DMatch >& matches,
              vector<Point2f>& points1,
              vector<Point2f>& points2
            );

void print_pts ( vector<Point2f>& points1,
                 vector<Point2f>& points2,
                 int start,
                 int end );

// 像素坐标转相机归一化坐标
Point2f pixel2cam ( const Point2f& p, const Mat& K );
void pixel2cam ( const double px, const double py, const Mat& K, double& cx, double& cy );
