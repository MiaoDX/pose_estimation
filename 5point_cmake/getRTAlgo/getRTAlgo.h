#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iomanip>
#include <fstream>
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
//typedef unsigned char BYTE;
//#define EXPORT_DEF  _declspec(dllexport)

//extern "C" EXPORT_DEF void _cdecl transformPoint(double H[9],double &x,double &y);
//extern "C" EXPORT_DEF bool _cdecl getHMatrix(char *featureName,char *name1,char *name2,double H[9],double sc);
//extern "C" EXPORT_DEF bool _cdecl getCameraPose(char *img1,char *img2,double K[9],double &rotate_x,double &rotate_y,double &rotate_z,
//				double &move_x,double &move_y,double &move_z,float scale = 1.0f,int ptsLimitbool = 3000, float ratio = 0.4,bool useGPU=false);
//extern "C" EXPORT_DEF void _cdecl SaveHMat(char *name1,char *name2,double H[9]);
//extern "C" EXPORT_DEF void _cdecl saveMatch(char *name);
//bool getHMatrix(char *featureName,const char *name1,const char *name2,double H[9],double _sc,vector<Point2f> &pts1,vector<Point2f> &pts2);


void extractMatchFeaturePoints(char* featurename, char* imagename1, char* imagename2, vector<Point2f> &pts1,vector<Point2f> &pts2, double max_ratio = 0.4, double scale = 1.0);

void unique_keypoint(vector<KeyPoint> &points);

void matchPointsRansac(vector<Point2f> &pts1,vector<Point2f> &pts2);

bool calculateRT(vector<Point2f> pts1,vector<Point2f> pts2, double K[9], 
	double &rotate_x,double &rotate_y,double &rotate_z, 
	double &move_x,double &move_y,double &move_z, int ptsLimit = 3000);

void getInvK(double invk[9],double K[9]);

void transformPoint(double H[9],double &x,double &y);