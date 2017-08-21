#include "pose_estimation_2d2d.h"
#include "getRTAlgo.h"

using namespace std;
using namespace cv;


int main ( int argc, char** argv )
{
    // Command Arguments : H:/projects/SLAM/python_code/dataset/our/trajs2/1.jpg H:/projects/SLAM/python_code/dataset/our/trajs2/4.jpg
    if ( argc != 3 )
    {
        cout << "usage: triangulation img1 img2" << endl;
        return 1;
    }
    //-- 读取图像
    //Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    //Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

    //Mat K = (Mat_<double> ( 3, 3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
	//K = np.array ( [[8607.8639, 0, 2880.72115], [0, 8605.4303, 1913.87935], [0, 0, 1]] ) # Canon5DMarkIII - EF50mm
	Mat K = (Mat_<double> ( 3, 3 ) << 8607.8639, 0, 2880.72115, 0, 8605.4303, 1913.87935, 0, 0, 1);
    double K_arr[9] = { 8607.8639, 0, 2880.72115, 0, 8605.4303, 1913.87935, 0, 0, 1 };

    vector<KeyPoint> keypoints_1_all, keypoints_2_all;
    vector<DMatch> matches;
    find_feature_matches ( img_1, img_2, keypoints_1_all, keypoints_2_all, matches );
	cout << "We found " << keypoints_1_all.size () << " key points in im1" << endl;
	cout << "We found " << keypoints_2_all.size () << " key points in im2" << endl;
    cout << "We found " << matches.size () << " pairs of points in total" << endl;
    //DebugMatchedKeyPoints ( img_1, img_2, keypoints_1_all, keypoints_2_all, matches );


    calcuateRT_test ( keypoints_1_all, keypoints_2_all, matches, K, K_arr );
    
    
    refineMatcheswithHomography ( keypoints_1_all, keypoints_2_all, matches );
    calcuateRT_test ( keypoints_1_all, keypoints_2_all, matches, K, K_arr );

    refineMatchesWithFundmentalMatrix ( keypoints_1_all, keypoints_2_all, matches );
    calcuateRT_test ( keypoints_1_all, keypoints_2_all, matches, K, K_arr );

    system ( "pause" );

    return 0;
}
