#include "pose_estimation.h"

using namespace std;
using namespace cv;


//void extractMatchFeaturePoints(char* featurename, char* imagename1, char* imagename2, vector<Point2f> &pts1,vector<Point2f> &pts2, double max_ratio = 0.4, double scale = 1.0);
void extractKeyPointsAndMatches ( string featurename, const string imagename1, const string imagename2,
    vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2, vector<DMatch>& matches, const bool withFlann = false );


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


void calcuateRT_test ( const vector<KeyPoint> kps1, const vector<KeyPoint> kps2, const vector<DMatch>& matches, const Mat& K);