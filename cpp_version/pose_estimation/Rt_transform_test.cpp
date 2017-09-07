#include "Rt_transform.h"


using namespace std;
using namespace cv;

/*
R = np.array([
0.99943541, 0.00186892, 0.03354636, -0.00198323, 0.99999234, 0.00337439,
-0.0335398, -0.00343902, 0.99943147
]).reshape(3, 3)

zyx_rad = GetEulerRadZYX(R)
print(zyx_rad.T)

zyx_degree = GetEulerDegreeZYX(R)
print(zyx_degree.T)

R2 = EulerZYXRad2R(zyx_rad)
print(R2)

R3 = EulerZYXDegree2R(zyx_degree)
print(R3)
 */

int main()
{

    Mat Rotation = (Mat_<double> ( 3, 3 ) << 0.99943541, 0.00186892, 0.03354636, -0.00198323, 0.99999234, 0.00337439,
                    -0.0335398, -0.00343902, 0.99943147);


    vector<double> zyx_rad = GetEulerRadZYX ( Rotation );
    cout << zyx_rad[0] << " " << zyx_rad[1] << " " << zyx_rad[2] << endl;

    vector<double> zyx_degree = GetEulerDegreeZYX ( Rotation );
    cout << zyx_degree[0] << " " << zyx_degree[1] << " " << zyx_degree[2] << endl;


    Mat R2 = EulerRadZYX2R ( zyx_rad[0], zyx_rad[1], zyx_rad[2] );
    cout << "R2:\n" << R2 << endl;

    Mat R3 = EulerDegreeZYX2R ( zyx_degree[0], zyx_degree[1], zyx_degree[2] );
    cout << "R3:\n" << R3 << endl;

    system ( "pause" );

}