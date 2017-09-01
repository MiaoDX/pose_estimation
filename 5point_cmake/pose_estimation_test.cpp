/*
 * This is our code to evaluate the pose_estimation, with real images.
 *
 * The refine part is worth evaluating, whether they are necessary or not.
 */


#include "pose_estimation.h"
#include "getRTAlgo.h"
using namespace std;
using namespace cv;

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc,char **argv)
{

    // Command Arguments : H:/projects/SLAM/python_code/dataset/our/trajs2/1.jpg H:/projects/SLAM/python_code/dataset/our/trajs2/4.jpg
    if ( argc != 3 ) {
        cout << "usage: pose_estimation_test img1 img2" << endl;
        return 1;
    }


    Mat K = (Mat_<double> ( 3, 3 ) << 8607.8639, 0, 2880.72115, 0, 8605.4303, 1913.87935, 0, 0, 1);


    //string featureName = "BRISK";
    //string featureName = "BRIEF";
    string featureName = "ORB";


    string imgname1 = argv[1];
    string imgname2 = argv[2];

    vector<KeyPoint> kpts1;
    vector<KeyPoint> kpts2;
    vector <DMatch> matches;
    //double max_ratio = 0.4;    // 0.4 or 0.6
    //double scale = 1.0;   // 1.0

    extractKeyPointsAndMatches ( featureName, imgname1, imgname2, kpts1, kpts2, matches);
    // extractKeyPointsAndMatches (featureName, imgname1, imgname2, kpts1, kpts2, matches, true);


    calcuateRT_test ( kpts1, kpts2, matches, K );

    // - REFINE
    refineMatcheswithHomography ( kpts1, kpts2, matches );
    calcuateRT_test ( kpts1, kpts2, matches, K );

    refineMatchesWithFundmentalMatrix ( kpts1, kpts2, matches );
    calcuateRT_test ( kpts1, kpts2, matches, K );

    system ( "pause" );
    return 0;
}
