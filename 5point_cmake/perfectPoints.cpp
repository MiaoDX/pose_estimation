/*
 * Copy from [relative-pose-estimation of prclibo](https://github.com/prclibo/relative-pose-estimation/blob/master/demo.cpp)
 * prclibo is the contributor of findEssentialMat of OpenCV 3.x and the test code is just simple and clean.
 * 
 * We also evaluate the code of [Nghia Ho's FIVE POINT ALGORITHM FOR ESSENTIAL MATRIX, 1 YEAR LATER …](http://nghiaho.com/?p=1675)
 * 
 * And print out potential R, t pair of both algorithms.
 * 
 * RESULTS:
 * 
 * In this experiment -- use the perfect points (project with problem, no noise), the findEssentialMat give promising results, while the 
 * Nghia Ho's can be not stable for the t (sometimes it get -t) and sometimes choose the wrong pairs.
 * 
 * However, the findEssentialMat can give us one and only t vector, while the Nghia Ho's give us potential pairs with different ts, so in 
 * real world, findEssentialMat can give us totally wrong answers. Experiments are set up otherwise.
 * 
 */

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
    Mat K = (Mat_<double> ( 3, 3 ) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);
    double K_arr[9] = { focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1 };

    //Mat K = (Mat_<double> ( 3, 3 ) << 8607.8639, 0, 2880.72115, 0, 8605.4303, 1913.87935, 0, 0, 1);
    //double K_arr[9] = { 8607.8639, 0, 2880.72115, 0, 8605.4303, 1913.87935, 0, 0, 1 };
    

    Mat rvec = (cv::Mat_<double>(3, 1) << 0.1 , 0.2 , 0.3);
    Mat tvec = (cv::Mat_<double>(3, 1) << 0.4 , 0.5 , 0.6);
    normalize(tvec, tvec);
    std::cout << "Expected rvec: " << rvec << std::endl;
    std::cout << "Expected tvec: " << tvec << std::endl;


    Mat rmat;
    Rodrigues(rvec, rmat);

    
    srand ( static_cast<int>(time ( 0 )) );
    RNG rng( rand () ); // use RNG rng() to avoid random values
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
     * Highly important, all the projected 2d points should be **normalized** 
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

    //vector<Point2f> ptsa = Mat_<Point2f>(x1s);	// [Mat, vector<point2f>，Iplimage等等常见类型转换](http://blog.csdn.net/foreverhehe716/article/details/6749175)
    vector<Point2f> pts1;
    vector<Point2f> pts2;
    for ( int i = 0; i < x1s.rows; i++ )
    {
        pts1.push_back ( Point2f ( x1s.row ( i ) ) );
        pts2.push_back ( Point2f ( x2s.row ( i ) ) );
    }


    // Mat F = cv::findFundamentalMat(x1s, x2s, noArray(), CV_RANSAC);
    Mat F = cv::findFundamentalMat ( pts1, pts2, noArray (), CV_RANSAC );
    cout << "F:" << endl << F << endl;

    

    Mat E_f_F = K.t() * F * K;
    cout << "E from F:" << endl << E_f_F << endl;    
    cout << "Scaled E from F:" << endl << scaled_E(E_f_F) << endl;

/*
 * Code below is almost the same as calculateRT_CV3 in getRTAlgo.cpp
 */
#ifdef _CV_VERSION_3
    // Mat E = findEssentialMat(x1s, x2s, K, CV_RANSAC, 0.99, 1, noArray());
    Mat E = findEssentialMat ( pts1, pts2, K, CV_RANSAC, 0.99, 1, noArray () );

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

    Mat R, t, r;
    recoverPose(E, x1s, x2s, K, R, t);
    DEBUG_RT ( R, t );
#else
    cout << "Seems we are not using OpenCV 3.x, so no findEssentialMat" << endl;
#endif


    std::cout << "=====================================================" << std::endl;
    std::cout << "Now, use Nghia Ho.'s algorithm" << std::endl;

    Mat R_5, t_5;
    calculateRT_5points ( pts1, pts2, K_arr, R_5, t_5, pts1.size () );
    DEBUG_RT ( R_5, t_5 );

    system("pause");
}
