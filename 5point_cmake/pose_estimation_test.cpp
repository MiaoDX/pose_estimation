/*
 * This is our code to evaluate the pose_estimation, with real images.
 */


#include "pose_estimation_2d2d.h"
#include "getRTAlgo.h"
using namespace std;

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc,char **argv){
    
  string imgname1 = "H:/projects/SLAM/python_code/dataset/our/trajs2/1.jpg";
  string imgname2 = "H:/projects/SLAM/python_code/dataset/our/trajs2/7b.jpg";

  Mat K = (Mat_<double> ( 3, 3 ) << 8607.8639, 0, 2880.72115, 0, 8605.4303, 1913.87935, 0, 0, 1);
  double K_arr[9] = { 8607.8639, 0, 2880.72115, 0, 8605.4303, 1913.87935, 0, 0, 1 };


  //char* featureName = "SIFT";
  string featureName = "BRISK";
  //char* featureName = "FAST";
  

  vector<Point2f> pts1;
  vector<Point2f> pts2;
  double max_ratio = 0.4;    // 0.4 or 0.6
  double scale = 1.0;   // 1.0

  extractMatchFeaturePoints(featureName, imgname1, imgname2, pts1, pts2, max_ratio, scale);

  matchPointsRansac(pts1, pts2);
  
  //-- 估计两张图像间运动
  Mat R, t;
  calculateRT_CV3 ( pts1, pts2, K, R, t );
  DEBUG_RT ( R, t );

  Mat R_5, t_5;
  calculateRT_5points ( pts1, pts2, K_arr, R_5, t_5, pts1.size () );
  DEBUG_RT ( R_5, t_5 );

  system ( "pause" );
  return 0;
}
