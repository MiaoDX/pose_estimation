#include "pose_estimation_2d2d.h"
#include "getRTAlgo.h"

using namespace std;
using namespace cv;

#define ORB_NFEATURES 1000

void find_feature_matches ( const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
#ifdef _CV_VERSION_3
    // used in OpenCV3 
    Ptr<FeatureDetector> detector = ORB::create ( ORB_NFEATURES );
#else
    initModule_nonfree ();
    initModule_features2d ();
    Ptr<FeatureDetector> detector = new ORB ( ORB_NFEATURES );
#endif

    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1, keypoints_1 );
    detector->detect ( img_2, keypoints_2 );

    find_feature_matches_from_keypoints ( img_1, img_2, keypoints_1, keypoints_2, matches );
}



void find_feature_matches_from_keypoints (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
#if _CV_VERSION_3
    // used in OpenCV3 
    Ptr<DescriptorExtractor> descriptor = ORB::create ( ORB_NFEATURES );
    
#else
    Ptr<DescriptorExtractor> descriptor = new ORB ( ORB_NFEATURES );

#endif

    //Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create ( "BruteForce-Hamming" );
    Ptr<DescriptorMatcher> matcher = new BFMatcher ( NORM_HAMMING , true);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < match.size(); i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < match.size (); i++ )
    {
        if ( match[i].distance <= max ( 2 * min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}


void DebugMatchedKeyPoints (
    const Mat& img_1, const Mat& img_2,
    const std::vector<KeyPoint>& keypoints_1,
    const std::vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches
)
{
    std::vector<KeyPoint> keypoints_1_matched;
    std::vector<KeyPoint> keypoints_2_matched;
    std::vector<Point2d> points_1_matched;
    std::vector<Point2d> points_2_matched;

    /* Debug */
    cout << "Output all matched keypoints and corresponding matches and their distance" << endl;
    for ( DMatch m : matches ) {
        keypoints_1_matched.push_back ( keypoints_1[m.queryIdx] );
        points_1_matched.push_back ( keypoints_1[m.queryIdx].pt );
        
        keypoints_2_matched.push_back ( keypoints_2[m.trainIdx] );
        points_2_matched.push_back ( keypoints_2[m.trainIdx].pt );
    }

    //-- Draw the descriptors
    Mat outimg1, outimg2;
    drawKeypoints ( img_1, keypoints_1, outimg1, Scalar::all ( -1 ), DrawMatchesFlags::DEFAULT );
    drawKeypoints ( img_2, keypoints_2, outimg2, Scalar::all ( -1 ), DrawMatchesFlags::DEFAULT );
    //imshow ( "Descriptors on im1", outimg1 );
    //imshow ( "Descriptors on im2", outimg2 );
    resize_and_show ( outimg1, 320, "Descriptors on im1" );
    resize_and_show ( outimg2, 320, "Descriptors on im2" );

    //-- 第五步:绘制匹配结果
    Mat img_match;
    //Mat img_goodmatch;
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, matches, img_match );
    cout << "-- All match num :" << matches.size () << endl;
    
    resize_and_show ( img_match, 320, "All matches" );    
}


Point2f pixel2cam ( const Point2f& p, const Mat& K )
{
	//[1、像素坐标与像平面坐标系之间的关系 ](http://blog.csdn.net/waeceo/article/details/50580607)
    return Point2f
    (
        (p.x - K.at<double> ( 0, 2 )) / K.at<double> ( 0, 0 ),
        (p.y - K.at<double> ( 1, 2 )) / K.at<double> ( 1, 1 )
    );
}




void calcuateRT_test ( const vector<KeyPoint> kps1, const vector<KeyPoint> kps2, const vector<DMatch>& matches, const Mat& K, double* K_arr )
{
    Mat R, t;
    vector<Point2f> points1, points2;
    kp2pts ( kps1, kps2, matches, points1, points2 );
    print_pts ( points1, points2, 0, 10 );
    calculateRT_CV3 ( points1, points2, K, R, t );
    DEBUG_RT ( R, t );

    Mat R_5, t_5;
    calculateRT_5points ( points1, points2, K_arr, R_5, t_5, points1.size () );
    DEBUG_RT ( R_5, t_5 );
}