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
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/gpu.hpp>
#endif


using namespace cv;
using namespace std;


//void extractMatchFeaturePoints(char* featurename, char* imagename1, char* imagename2, vector<Point2f> &pts1,vector<Point2f> &pts2, double max_ratio = 0.4, double scale = 1.0);
void extractMatchFeaturePoints ( string featurename, string imagename1, string imagename2, vector<Point2f> &pts1, vector<Point2f> &pts2, double max_ratio = 0.4, double scale = 1.0 );

void unique_keypoint(vector<KeyPoint> &points);

void matchPointsRansac(vector<Point2f> &pts1,vector<Point2f> &pts2);

bool calculateRT_5points ( const vector<Point2f> vpts1, const vector<Point2f> vpts2, double K[9], Mat& R, Mat& t, int ptsLimit, bool showAll = true );

bool calculateRT_5points (vector<Point2f> pts1,vector<Point2f> pts2, double K[9],
	double &rotate_x,double &rotate_y,double &rotate_z, 
	double &move_x,double &move_y,double &move_z, int ptsLimit = 3000);

void calculateRT_CV3 (
    const vector<Point2f> points1,
    const vector<Point2f> points2,
    const Mat K,
    Mat& R, Mat& t );

void getInvK(double invk[9],double K[9]);

void transformPoint(double H[9],double &x,double &y);

void resize_and_show ( const Mat& im, int target_height = 640, string name = "Image" );

Mat scaled_E ( const Mat& E );

void rotate_angle ( const Mat& R );

void DEBUG_RT ( const Mat& R, const Mat& t );

// [Copy from SFM](https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/fundamental.cpp)
void essentialFromFundamental ( const Mat &F,
    const Mat &K1,
    const Mat &K2,
    Mat& E );