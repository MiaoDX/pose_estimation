#include "getRTAlgo.h"
#include "5point.h"


using namespace cv;
using namespace std;

/*保存的参考图片的SIFT信息*/
/*
int height = 1624;
int width = 1224;
int channels = 4;
int image_bytes_count ;
static vector<Point2f> mpts1;
static vector<Point2f> mpts2;
*/



void extractKeyPointsAndMatches ( string featurename, const string imagename1, const string imagename2,
    vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2, vector<DMatch>& matches, const bool withFlann)
{
    kpts1.clear (); kpts2.clear (); matches.clear ();
    
    // https://stackoverflow.com/questions/313970/how-to-convert-stdstring-to-lower-case
    transform ( featurename.begin (), featurename.end (), featurename.begin (), ::toupper );
	
	cout << "Feature name:" << featurename << endl;
	
	Mat img1 = imread(string( imagename1 ), 0);
	Mat img2 = imread(string( imagename2 ), 0);
	        
    Ptr<FeatureDetector> fd; Ptr<DescriptorExtractor> de;
    getFeatureDetectorDescriptorExtractor ( fd, de, featurename );

    Mat dsp1, dsp2;
    extractFeaturesAndDescriptors ( fd, de, img1, img2, kpts1, kpts2, dsp1, dsp2 );
	cout << "KeyPoints num. kpts1.size:" << kpts1.size () << ", kpts2.size:" << kpts2.size () << endl;


    Ptr<DescriptorMatcher> matcher;
    int normType;
    if (featurename == "ORB" || featurename == "BRIEF" || featurename == "BRISK") // needs HAMMING
    {
        cout << "Going to use NORM_HAMMING, like ORB" << endl;
        normType = NORM_HAMMING;
    }
    else
    {
        cout << "Going to use NORM_L2, like SIFT" << endl;
        normType = NORM_L2;
    }
    if (withFlann)
    {
        cout << "Going to use withFlann" << endl;
        matcher = getMatchTypeFlann (normType);
    }
    else
    {
        matcher = getMatchTypeNormal ( normType );
    }

    if (normType == NORM_HAMMING)
    {
        match_with_NORM_HAMMING ( matcher, dsp1, dsp2, matches );
    }
    else
    {
        match_with_knnMatch ( matcher, dsp1, dsp2, matches );
    }


	cout << "Matches num:" << matches.size () << endl;
}









bool calculateRT_5points ( const vector<Point2f>& vpts1, const vector<Point2f>& vpts2, const Mat& K, Mat& R, Mat& t, int ptsLimit, bool withDebug )
{
    int npts = static_cast<int>(vpts1.size ());
    if ( npts < 5 ) return false;
    int chosenNum = min ( npts, ptsLimit );

    if (withDebug)
    {
        cout << "In calculateRT_5points, num of points: " << npts << ", chosenNum:" << chosenNum << endl;
    }
    


    //pixel2cam
    vector<double> _pts1_cam, _pts2_cam;
    _pts1_cam.resize ( chosenNum * 2 ); _pts2_cam.resize ( chosenNum * 2 );

    for(int i = 0; i < chosenNum; i++ )
    {
        pixel2cam ( vpts1[i].x, vpts1[i].y, K, _pts1_cam[i * 2], _pts1_cam[i * 2 + 1] );
        pixel2cam ( vpts2[i].x, vpts2[i].y, K, _pts2_cam[i * 2], _pts2_cam[i * 2 + 1] );
        //printf("(%g, %g) -> (%g, %g)\n", vpts1[i].x, vpts1[i].y, _pts1_cam[2*i], _pts1_cam[2*i+1]);
        //printf ( "(%g, %g) -> (%g, %g)\n", vpts2[i].x, vpts2[i].y, _pts2_cam[2 * i], _pts2_cam[2 * i + 1] );
    }


    vector <cv::Mat> E; // essential matrix
    vector <cv::Mat> P;
    vector<int> inliers;

    bool ret = Solve5PointEssential ( _pts1_cam.data(), _pts2_cam.data(), chosenNum, E, P, inliers ); // 从4个解得到1个最优解；P：映射矩阵 [R|t]

    if (withDebug)
    {
        cout << "============== Solve5PointEssential =============" << endl;
        printf ( "Solve5PointEssential() found %llu solutions:\n", E.size () );
    }
    
    size_t best_index = -1;
    if ( ret ) {
        for ( size_t i = 0; i < E.size (); i++ ) {
            if ( cv::determinant ( P[i] ( cv::Range ( 0, 3 ), cv::Range ( 0, 3 ) ) ) < 0 ) P[i] = -P[i];
            
            if( withDebug )
            {
                R = P[i].colRange ( 0, 3 );
                t = P[i].colRange ( 3, 4 );
                printf ( "%zd/%zd : %d/%d\t", i, E.size(), inliers[i], npts );
                DEBUG_RT ( R, t );
            }
            
            if ( best_index == -1 || inliers[best_index] < inliers[i] ) best_index = i;
        }
    }
    else {
        cout << "Could not find a valid essential matrix" << endl;
        return false;
    }
    if (withDebug)
    {
        cout << "============== Solve5PointEssential =============" << endl;
    }

    cv::Mat p_mat = P[best_index];
    cv::Mat Ematrix = E[best_index];

    R = p_mat.colRange ( 0, 3 );
    t = p_mat.colRange ( 3, 4 );

    return true;
}


bool calculateRT_5points ( const vector<Point2f>& pts1, const vector<Point2f>& pts2, double K[9],
    double &rotate_x, double &rotate_y, double &rotate_z,
    double &move_x, double &move_y, double &move_z, int ptsLimit)
{


    Mat R, t;
    Mat k_M ( 3, 3, CV_64FC1, K );
    calculateRT_5points ( pts1, pts2, k_M, R, t, ptsLimit, true );

	double rot_x,rot_y,rot_z;
	rot_y = asin(R.at<double>(2,0));
	rot_z = asin(-R.at<double>(1,0)/cos(rot_y));
	rot_x = asin(-R.at<double>(2,1)/cos(rot_y));
	rotate_x = rot_x*180/CV_PI;
	rotate_y = rot_y*180/CV_PI;
	rotate_z = rot_z*180/CV_PI;

	move_x = t.at<double>(0,0);
	move_y = t.at<double>(1,0);
	move_z = t.at<double>(2,0);
	
    return true;
}






void calculateRT_CV3 (
    const vector<Point2f> points1,
    const vector<Point2f> points2,
    const Mat K,
    Mat& R, Mat& t,
    bool withDebug)
{
    assert ( points1.size () > 0 && points1.size () == points2.size () && K.size () == Size ( 3, 3 ) );
    R.release (); t.release ();

#ifndef _CV_VERSION_3
    cout << "Seems we are not using OpenCV 3.x, so no findEssentialMat, just return." << endl;
    return;
#else

    //-- 计算本质矩阵
    Mat E = findEssentialMat ( points1, points2, K );
    
    
    if (withDebug)
    {
        cout << "E from findEssentialMat:" << endl << E << endl;
        Mat E_scaled = scaled_E ( E );
        cout << "Scaled E:" << endl << E_scaled << endl;

        // we can get four potential answers here
        Mat R1_5pt, R2_5pt, tvec_5pt, rvec1_5pt, rvec2_5pt;
        decomposeEssentialMat ( E, R1_5pt, R2_5pt, tvec_5pt );
        cout << "============== decomposeEssentialMat =============" << endl;
        DEBUG_RT ( R1_5pt, tvec_5pt );
        DEBUG_RT ( R2_5pt, tvec_5pt );
        cout << "============== decomposeEssentialMat =============" << endl;
    }
    

    //vector<uchar> inliersMask ( points1.size () ); // we can not set as this, since it seems that the Mask is changed shape in the method
    Mat inliersMask;

    //-- 从本质矩阵中恢复旋转和平移信息.re
    recoverPose ( E, points1, points2, K, R, t, inliersMask );
    
    // cout << "inliersMask, channels:" << inliersMask.channels () << ", type:" << inliersMask.type () << ", size:" << inliersMask.size() << endl;

    
    vector<Point2f> inliers_pts1, inliers_pts2;
    
    for ( int i = 0; i < inliersMask.rows; i++ ) {
        if ( inliersMask.at<uchar>(i, 0) )
        {
            inliers_pts1.push_back ( points1[i] );
            inliers_pts2.push_back ( points2[i] );
        }
    }
    
   

    if (withDebug)
    {
        cout << "In recoverPose, points:" << points1.size () << "->" << inliers_pts1.size () << endl;
    }


    
#endif
}


void calcuateRT_test ( const vector<KeyPoint> kps1, const vector<KeyPoint> kps2, const vector<DMatch>& matches, const Mat& K)
{
    Mat R, t;
    vector<Point2f> points1, points2;
    kp2pts ( kps1, kps2, matches, points1, points2 );
    print_pts ( points1, points2, 0, 10 );
    calculateRT_CV3 ( points1, points2, K, R, t );
    DEBUG_RT ( R, t );

    Mat R_5, t_5;
    calculateRT_5points ( points1, points2, K, R_5, t_5, points1.size () );
    DEBUG_RT ( R_5, t_5 );
}