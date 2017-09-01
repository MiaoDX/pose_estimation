#include "pose_estimation_utils.h"

using namespace std;
using namespace cv;


Mat scaled_E ( const Mat& E )
{
    Mat scaled_E = E / E.at<double> ( 2, 2 );
    //cout << "Scaled E:" << scaled_E << endl;
    return scaled_E;
}

void rotate_angle ( const Mat& R )
{
    double r11 = R.at<double> ( 0, 0 ), r21 = R.at<double> ( 1, 0 ), r31 = R.at<double> ( 2, 0 ), r32 = R.at<double> ( 2, 1 ), r33 = R.at<double> ( 2, 2 );

    //计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
    //旋转顺序为z、y、x

    const double PI = 3.14159265358979323846;
    double thetaz = atan2 ( r21, r11 ) / PI * 180;
    double thetay = atan2 ( -1 * r31, sqrt ( r32*r32 + r33*r33 ) ) / PI * 180;
    double thetax = atan2 ( r32, r33 ) / PI * 180;

    cout << "thetaz:" << thetaz << " thetay:" << thetay << " thetax:" << thetax << endl;
}

void DEBUG_RT ( const Mat& R, const Mat& t )
{
    if(R.empty() )
    {
        cout << "Seems R is empty in DEBUG_RT, just return." << endl;
        return;
    }

    Mat r;
    cv::Rodrigues ( R, r ); // r为旋转向量形式，用Rodrigues公式转换为矩阵

    //cout << "R=" << endl << R << endl;
    //cout << "r=" << endl << r << endl;
    rotate_angle ( R );
    cout << "t:" << t.t() << endl;
}

void essentialFromFundamental ( const Mat &F,
    const Mat &K1,
    const Mat &K2,
    Mat& E )
{
    E = K2.t () * F * K1;
}
