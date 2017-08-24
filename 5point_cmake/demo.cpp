#include <algorithm>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "getRTAlgo.h"

using namespace cv;
using namespace std;

int main()
{
    double N = 100;
    double bound_2d = 5;

    double focal = 300;
    Point2d pp(0, 0);
    //Mat K = (Mat_<double> ( 3, 3 ) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);
    //double K_arr[9] = { focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1 };

    Mat K = (Mat_<double> ( 3, 3 ) << 8607.8639, 0, 2880.72115, 0, 8605.4303, 1913.87935, 0, 0, 1);
    double K_arr[9] = { 8607.8639, 0, 2880.72115, 0, 8605.4303, 1913.87935, 0, 0, 1 };


    Mat rvec = (cv::Mat_<double>(3, 1) << 0.1 , 0.2 , 0.3);
    Mat tvec = (cv::Mat_<double>(3, 1) << 0.4 , 0.5 , 0.6);
    normalize(tvec, tvec);
    std::cout << "Expected rvec: " << rvec << std::endl;
    std::cout << "Expected tvec: " << tvec << std::endl;


    Mat rmat;
    Rodrigues(rvec, rmat);

    
    srand ( static_cast<int>(time ( 0 )) );
    RNG rng( rand () );
    Mat Xs(N, 3, CV_64F);
    rng.fill(Xs, RNG::UNIFORM, -bound_2d, bound_2d);

    cout << "Print out the first five lines of random values:" << endl;
    for(int i = 0; i < 5; i++ )
    {
        cout << Xs.row (i) << endl;
    }

    Mat x1s = K * Xs.t();
    Mat x2s = rmat * Xs.t();
    for (int j = 0; j < x2s.cols; j++) x2s.col(j) += tvec;
    x2s = K * x2s;

    /*
     * Highly important, all the 3d points should be **normalized** and project to 2d
     */
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

    Mat F = cv::findFundamentalMat(x1s, x2s, noArray(), CV_RANSAC);
    cout << "F:" << endl << F << endl;

    Mat E_f_F = K.t() * F * K;
    cout << "E from F:" << endl << E_f_F << endl;    
    cout << "Scaled E from F:" << endl << scaled_E(E_f_F) << endl;


#ifdef _CV_VERSION_3
    // Mat E = findEssentialMat(x1s, x2s, focal, pp, CV_RANSAC, 0.99, 1, noArray() ); 
    Mat E = findEssentialMat(x1s, x2s, K, CV_RANSAC, 0.99, 1, noArray());

    std::cout << "=====================================================" << std::endl;
    cout << "E from findEssentialMat:" << endl << E << endl;
    cout << "Scaled E:" << endl << scaled_E ( E ) << endl;
    

    
    // we can get four potential answers here
    Mat R1_5pt, R2_5pt, tvec_5pt, rvec1_5pt, rvec2_5pt; 
    decomposeEssentialMat(E, R1_5pt, R2_5pt, tvec_5pt); 
    cout << "============== decomposeEssentialMat =============" << endl;
    DEBUG_RT ( R1_5pt, tvec_5pt );
    DEBUG_RT ( R2_5pt, tvec_5pt );
    cout << "============== decomposeEssentialMat =============" << endl;

    /*
    Rodrigues(R1_5pt, rvec1_5pt); 
    Rodrigues(R2_5pt, rvec2_5pt); 
    std::cout << "5-pt-nister rvec: " << std::endl; 
    std::cout << rvec1_5pt << std::endl; 
    std::cout << rvec2_5pt << std::endl; 
    std::cout << "5-pt-nister tvec: " << std::endl; 
    std::cout << tvec_5pt << std::endl; 
    std::cout << -tvec_5pt << std::endl; 
    */

    Mat R, t, r;
    recoverPose(E, x1s, x2s, K, R, t);
    //cv::Rodrigues(R, r); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    //std::cout << "R=" << std::endl << R << std::endl;
    //std::cout << "r=" << std::endl << r << std::endl;
    //std::cout << "t is " << std::endl << t << std::endl;
    DEBUG_RT ( R, t );

#endif


    std::cout << "=====================================================" << std::endl;
    std::cout << "Now, use Nghia Ho.'s algorithm to have a try" << std::endl;

    //vector<Point2f> ptsa = Mat_<Point2f>(x1s);	// [Mat, vector<point2f>，Iplimage等等常见类型转换](http://blog.csdn.net/foreverhehe716/article/details/6749175)
    vector<Point2f> pts1;
    vector<Point2f> pts2;
    for ( int i = 0; i < x1s.rows; i++ )
    {
        pts1.push_back ( Point2f ( x1s.row ( i ) ) );
        pts2.push_back ( Point2f ( x2s.row ( i ) ) );
    }

    Mat R_5, t_5;
    calculateRT ( pts1, pts2, K_arr, R_5, t_5, pts1.size () );
    DEBUG_RT ( R_5, t_5 );

    /*
    std::cout << "=====================================================" << std::endl;
    std::cout << "Nghia Ho.'s algorithm again" << std::endl;
 

    double rotate_x, rotate_y, rotate_z, move_x, move_y, move_z;
    calculateRT ( pts1, pts2, K_arr, rotate_x, rotate_y, rotate_z, move_x, move_y, move_z, N );
    cout << "The results are:" << endl;
    cout << "rotate_x:" << rotate_x << endl;
    cout << "rotate_y:" << rotate_y << endl;
    cout << "rotate_z:" << rotate_z << endl;
    cout << "move_x:" << move_x << endl;
    cout << "move_y:" << move_y << endl;
    cout << "move_z:" << move_z << endl;
    */

    system("pause");
}
