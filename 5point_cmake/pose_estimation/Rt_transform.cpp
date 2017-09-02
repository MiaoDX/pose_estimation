/*
 * R,t and Euler angle transform
 *
 * https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp
 */


#include "Rt_transform.h"
#include <opencv2/stitching/detail/util_inl.hpp>

using namespace std;
using namespace cv;

#define M_PI 3.141592653589793


Mat RPY(double roll,double pitch,double yaw)
{
    double ca1,cb1,cc1,sa1,sb1,sc1;
    ca1 = cos(yaw);
    sa1 = sin(yaw);
    cb1 = cos(pitch);
    sb1 = sin(pitch);
    cc1 = cos(roll);
    sc1 = sin(roll);

    Mat Rotation = (Mat_<double> ( 3, 3 ) << ca1*cb1,ca1*sb1*sc1 - sa1*cc1,ca1*sb1*cc1 + sa1*sc1,
                    sa1*cb1,sa1*sb1*sc1 + ca1*cc1,sa1*sb1*cc1 - ca1*sc1,
                    -sb1,cb1*sc1,cb1*cc1);

    return Rotation;
}

// Gives back a rotation matrix specified with RPY convention
vector<double> GetRPY ( const Mat R )
{

    assert ( R.size () == Size ( 3, 3 ) );

    double data0 = R.at<double> ( 0, 0 );
    double data1 = R.at<double> ( 0, 1 );
    double data3 = R.at<double> ( 1, 0 );
    double data4 = R.at<double> ( 1, 1 );
    double data6 = R.at<double> ( 2, 0 );
    double data7 = R.at<double> ( 2, 1 );
    double data8 = R.at<double> ( 2, 2 );

    double roll, pitch, yaw;

    double epsilon = 1E-12;
    pitch = atan2 ( -data6, sqrt ( detail::sqr ( data0 ) + detail::sqr ( data3 ) ) );

    if ( fabs ( pitch ) > (M_PI / 2.0 - epsilon) ) {
        yaw = atan2 ( -data1, data4 );
        roll = 0.0;
    }
    else {
        roll = atan2 ( data7, data8 );
        yaw = atan2 ( data3, data0 );
    }

    vector<double> RPY { roll, pitch, yaw };


    return RPY;
}



/**  EulerZYX constructs a Rotation from the Euler ZYX parameters:
*   -  First rotate around Z with alfa,
*   - then around the new Y with beta,
*   - then around new X with gamma.
*
*  Closely related to RPY-convention.
*
*  Invariants:
*  	- EulerZYX(alpha,beta,gamma) == EulerZYX(alpha +/- PI, PI-beta, gamma +/- PI)
*  	- (angle + 2*k*PI)
**/
Mat EulerRadZYX2R ( double Alfa, double Beta, double Gamma )
{
    return RPY ( Gamma, Beta, Alfa );
}


Mat EulerDegreeZYX2R ( double Alfa, double Beta, double Gamma )
{
    Alfa = Alfa / 180 * M_PI;
    Beta = Beta / 180 * M_PI;
    Gamma = Gamma / 180 * M_PI;
    return RPY ( Gamma, Beta, Alfa );
}

/**   GetEulerZYX gets the euler ZYX parameters of a rotation :
*   First rotate around Z with alfa,
*   then around the new Y with beta, then around
*   new X with gamma.
*
*  Range of the results of GetEulerZYX :
*  -  -PI <= alfa <= PI
*  -   -PI <= gamma <= PI
*  -  -PI/2 <= beta <= PI/2
*
*  if beta == PI/2 or beta == -PI/2, multiple solutions for gamma and alpha exist.  The solution where gamma==0
*  is chosen.
*
*
*  Invariants:
*  	- EulerZYX(alpha,beta,gamma) == EulerZYX(alpha +/- PI, PI-beta, gamma +/- PI)
*  	- and also (angle + 2*k*PI)
*
*  Closely related to RPY-convention.
**/
vector<double> GetEulerRadZYX ( const Mat& R)
{
    //GetRPY ( Gamma, Beta, Alfa );
    vector<double> RPY = GetRPY ( R );
    double Gamma = RPY[0], Beta = RPY[1], Alfa = RPY[2];
    return vector<double> {Alfa, Beta, Gamma};
}

vector<double> GetEulerDegreeZYX ( const Mat& R )
{
    vector<double> zyx_rad = GetEulerRadZYX ( R );
    return vector<double> {zyx_rad[0]/M_PI*180, zyx_rad[1] / M_PI * 180, zyx_rad[2] / M_PI * 180, };
}