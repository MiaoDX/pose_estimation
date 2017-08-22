/*
 * This is our code to evaluate the pose_estimation, with real images.
 * 
 * The refine part is worth evaluating, whether they are necessary or not.
 */


#include "pose_estimation_2d2d.h"
#include "getRTAlgo.h"
using namespace std;

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc,char **argv){
    
    // Command Arguments : H:/projects/SLAM/python_code/dataset/our/trajs2/1.jpg H:/projects/SLAM/python_code/dataset/our/trajs2/4.jpg
    if ( argc != 3 )
    {
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

  // - REFINE
  refineMatcheswithHomography ( kpts1, kpts2, matches );
  // refineMatchesWithFundmentalMatrix ( kpts1, kpts2, matches );
    
  vector<Point2f> pts1, pts2;
  kp2pts ( kpts1, kpts2, matches, pts1, pts2 );

  // matchPointsRansac(pts1, pts2);
  
  //-- 估计两张图像间运动
  Mat R, t;
  calculateRT_CV3 ( pts1, pts2, K, R, t );
  DEBUG_RT ( R, t );

  Mat R_5, t_5;
  calculateRT_5points ( pts1, pts2, K, R_5, t_5, pts1.size () );
  DEBUG_RT ( R_5, t_5 );

  system ( "pause" );
  return 0;
}
