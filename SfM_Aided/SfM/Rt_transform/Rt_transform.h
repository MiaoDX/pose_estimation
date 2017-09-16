#ifndef RT_TRANSFORM_H
#define RT_TRANSFORM_H



#include <vector>

#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

cv::Mat EulerRadZYX2R ( double Alfa, double Beta, double Gamma );
cv::Mat EulerDegreeZYX2R ( double Alfa, double Beta, double Gamma );
vector<double> GetEulerRadZYX ( const cv::Mat& R );
vector<double> GetEulerDegreeZYX ( const cv::Mat& R );

vector<double> rotate_angle ( const cv::Mat& R );

#endif