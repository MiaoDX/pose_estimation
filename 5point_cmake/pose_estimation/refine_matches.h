#pragma once

#include "pose_estimation.h"

using namespace std;
using namespace cv;


void refineMatcheswithHomography ( const vector<KeyPoint> kps1, const vector<KeyPoint> kps2,
                                   vector<DMatch>& matches, const double reprojectionThreshold = 3.0);

void refineMatchesWithFundmentalMatrix ( const vector<KeyPoint> kps1, const vector<KeyPoint> kps2,
        vector<DMatch>& matches );


void unique_keypoint(vector<KeyPoint> &points);

void matchPointsRansac(vector<Point2f> &pts1,vector<Point2f> &pts2);