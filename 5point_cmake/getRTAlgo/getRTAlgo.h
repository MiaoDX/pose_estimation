#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // toupper
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef _CV_VERSION_3
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#else
#include <opencv2/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/gpu.hpp>
#endif


using namespace cv;
using namespace std;


//void extractMatchFeaturePoints(char* featurename, char* imagename1, char* imagename2, vector<Point2f> &pts1,vector<Point2f> &pts2, double max_ratio = 0.4, double scale = 1.0);
void extractKeyPointsAndMatches ( string featurename, const string imagename1, const string imagename2,
    vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2, vector<DMatch>& matches, const bool withFlann = false );

void getFeatureDetectorDescriptorExtractor (Ptr<FeatureDetector>& fd, Ptr<DescriptorExtractor>& de, const string featurename = "SIFT");

void extractFeaturesAndDescriptors (const Ptr<FeatureDetector>& fd, const Ptr<DescriptorExtractor>& de , const Mat& im1, const Mat& im2, 
    vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2, Mat& dsp1, Mat& dsp2 );

Ptr<DescriptorMatcher> getMatchTypeNormal ( const int normType = NORM_L2 );

Ptr<DescriptorMatcher> getMatchTypeFlann ( const int normType = NORM_L2 );

void match_with_NORM_HAMMING ( const Ptr<DescriptorMatcher>& matcher, const Mat& des1, const Mat& des2, 
    vector<DMatch>& matches, double threshold_dis = 30.0 );

void match_with_knnMatch ( const Ptr<DescriptorMatcher>& matcher, const Mat& des1, const Mat& des2, 
    vector<DMatch>& matches, float minRatio = 1.f / 1.5f );

void refineMatcheswithHomography ( const vector<KeyPoint> kps1, const vector<KeyPoint> kps2,
    vector<DMatch>& matches, const double reprojectionThreshold = 3.0);

void refineMatchesWithFundmentalMatrix ( const vector<KeyPoint> kps1, const vector<KeyPoint> kps2, 
    vector<DMatch>& matches );




void unique_keypoint(vector<KeyPoint> &points);

void matchPointsRansac(vector<Point2f> &pts1,vector<Point2f> &pts2);


bool calculateRT_5points ( const vector<Point2f>& vpts1, const vector<Point2f>& vpts2, const Mat& K, Mat& R, Mat& t, int ptsLimit, bool withDebug = true );


bool calculateRT_5points ( const vector<Point2f>& pts1, const vector<Point2f>& pts2, double K[9],
	double &rotate_x,double &rotate_y,double &rotate_z, 
	double &move_x,double &move_y,double &move_z, int ptsLimit = 3000);



void calculateRT_CV3 (
    const vector<Point2f> points1,
    const vector<Point2f> points2,
    const Mat K,
    Mat& R, Mat& t,
    bool withDebug = true);

void resize_and_show ( const Mat& im, int target_height = 640, string name = "Image" );

Mat scaled_E ( const Mat& E );

void rotate_angle ( const Mat& R );

void DEBUG_RT ( const Mat& R, const Mat& t );

// [Copy from SFM](https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/fundamental.cpp)
void essentialFromFundamental ( const Mat &F,
    const Mat &K1,
    const Mat &K2,
    Mat& E );

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