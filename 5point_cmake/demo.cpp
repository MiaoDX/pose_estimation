#include <algorithm>
#include <opencv2/opencv.hpp>
//#include "../five-point-nister/five-point.hpp"

#include "5point.h"
#include "getRTAlgo.h"

using namespace cv; 
using namespace std;

int main()
{
    double N = 50; 
    double bound_2d = 5; 

    double focal = 300; 
    Point2d pp(0, 0); 
    
    Mat rvec = (cv::Mat_<double>(3, 1) << 0.1, 0.2, 0.3); 
    Mat tvec = (cv::Mat_<double>(3, 1) << 0.4, 0.5, 0.6); 
    normalize(tvec, tvec); 
    std::cout << "Expected rvec: " << rvec << std::endl; 
    std::cout << "Expected tvec: " << tvec << std::endl; 


    Mat rmat; 
    Rodrigues(rvec, rmat); 

    Mat K = (Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1); 
    
    RNG rng; 
    Mat Xs(N, 3, CV_64F); 
    rng.fill(Xs, RNG::UNIFORM, -bound_2d, bound_2d); 

	Mat x1s = K * Xs.t(); 
    Mat x2s = rmat * Xs.t(); 
    for (int j = 0; j < x2s.cols; j++) x2s.col(j) += tvec; 
    x2s = K * x2s; 

    x1s.row(0) /= x1s.row(2); 
    x1s.row(1) /= x1s.row(2); 
    x1s.row(2) /= x1s.row(2); 

    x2s.row(0) /= x2s.row(2); 
    x2s.row(1) /= x2s.row(2); 
    x2s.row(2) /= x2s.row(2); 
    
    x1s = x1s.t(); 
    x2s = x2s.t(); 
    
    x1s = x1s.colRange(0, 2) * 1.0; 
    x2s = x2s.colRange(0, 2) * 1.0; 

 
#ifdef _CV_VERSION_3
    Mat E = findEssentialMat(x1s, x2s, focal, pp, CV_RANSAC, 0.99, 1, noArray() ); 
    std::cout << "=====================================================" << std::endl; 
    Mat R1_5pt, R2_5pt, tvec_5pt, rvec1_5pt, rvec2_5pt; 
    decomposeEssentialMat(E, R1_5pt, R2_5pt, tvec_5pt); 
    Rodrigues(R1_5pt, rvec1_5pt); 
    Rodrigues(R2_5pt, rvec2_5pt); 
    std::cout << "5-pt-nister rvec: " << std::endl; 
    std::cout << rvec1_5pt << std::endl; 
    std::cout << rvec2_5pt << std::endl; 
    std::cout << "5-pt-nister tvec: " << std::endl; 
    std::cout << tvec_5pt << std::endl; 
    std::cout << -tvec_5pt << std::endl; 


	Mat R, t, r;
	recoverPose ( E, x1s, x2s, K, R, t );
	cv::Rodrigues ( R, r ); // r为旋转向量形式，用Rodrigues公式转换为矩阵
	std::cout << "R=" << std::endl << R << std::endl;
	std::cout << "r=" << std::endl << r << std::endl;
	std::cout << "t is " << std::endl << t << std::endl;
#endif



	std::cout << "Now, use Nghia Ho.'s algorithm to have a try" << std::endl;
	// bool ret = Solve5PointEssential ( pts1, pts2, num_pts, E, P, inliers );
	//bool calculateRT ( vector<Point2f> vpts1, vector<Point2f> vpts2, double K[9],
	//	double &rotate_x, double &rotate_y, double &rotate_z,
	//	double &move_x, double &move_y, double &move_z, int ptsLimit )

	double rotate_x, rotate_y, rotate_z, move_x, move_y, move_z;

	double K_arr[9] = { focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1 };

	//vector<Point3f> ptsa = Mat_<Point3f>(x1s);	// [Mat, vector<point2f>，Iplimage等等常见类型转换](http://blog.csdn.net/foreverhehe716/article/details/6749175)
	//vector<Point2f> pts1 = Mat_<Point2f> ( x1s );
	//vector<Point2f> pts2 = Mat_<Point2f>(x2s);

	vector<Point2f> pts1;
	vector<Point2f> pts2;
	for ( int i = 0; i < x1s.rows; i++ ) {
		pts1.push_back ( Point2f ( x1s.row ( i ) ) );
		pts2.push_back ( Point2f ( x2s.row ( i ) ) );
	}


	calculateRT ( pts1, pts2, K_arr, rotate_x, rotate_y, rotate_z, move_x, move_y, move_z, N );
	
	cout << "The results are:" << endl;
	cout << "rotate_x:" << rotate_x << endl;
	cout << "rotate_y:" << rotate_y << endl;
	cout << "rotate_z:" << rotate_z << endl;
	cout << "move_x:" << move_x << endl;
	cout << "move_y:" << move_y << endl;
	cout << "move_z:" << move_z << endl;

	system("pause");
}
