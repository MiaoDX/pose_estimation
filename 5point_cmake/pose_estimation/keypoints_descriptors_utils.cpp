#include "keypoints_descriptors_utils.h"

using namespace std;
using namespace cv;


void getFeatureDetectorDescriptorExtractor ( Ptr<FeatureDetector>& fd, Ptr<DescriptorExtractor>& de, const string featurename)
{
    fd.release ();
    de.release ();
    // fd->clear (); de->clear (); // not able to clear

#ifdef _CV_VERSION_3

    if ( featurename == "BRISK" ) {
        fd = de = BRISK::create ( 30, 3, 1.0F ); // These are now the default values
    }
    else if ( featurename == "BRIEF" ) {   // http://docs.opencv.org/3.2.0/dc/d7d/tutorial_py_brief.html
        fd = xfeatures2d::StarDetector::create(); // It take tons of time
        de = xfeatures2d::BriefDescriptorExtractor::create ();
    }
    else if ( featurename == "ORB" ) {
        fd = de = ORB::create ( 2000 ); // the default nfeatures of 500 is too small, 50000 seems nice
    }


    else if ( featurename == "FAST" ) {
        fd = FastFeatureDetector::create ( 40, true ); // it is not default 10
        de = xfeatures2d::FREAK::create ();
    }
    else if ( featurename == "SURF" ) {
        fd = de = xfeatures2d::SURF::create ();
    }
    else {    //SIFT
        fd = de = xfeatures2d::SIFT::create ();
    }
#else
    //在执行提取特征向量函数之前，必须执行该函数~！！
    initModule_nonfree ();
    initModule_features2d ();

    if ( featurename == "BRISK" ) {
        fd = de = new BRISK ( 30, 3, 1.0F ); // These are now the default values
    }
    else if ( featurename == "BRIEF" ) {   // http://docs.opencv.org/3.2.0/dc/d7d/tutorial_py_brief.html
        fd = new StarDetector ();
        de = new BriefDescriptorExtractor ();
    }
    else if ( featurename == "ORB" ) {
        fd = de = new ORB ( 2000 ); // These are now the default values
    }


    else if ( featurename == "FAST" ) {
        fd = new FastFeatureDetector ( 40, true ); // it is not default 10
        de = new FREAK ();
    }
    else if ( featurename == "SURF" ) {
        fd = de = new SURF ();
    }
    else {    //SIFT
        fd = de = new SIFT ();
    }
#endif // _CV_VERSION_3


}

void extractFeaturesAndDescriptors ( const Ptr<FeatureDetector>& fd, const Ptr<DescriptorExtractor>& de, const Mat& im1, const Mat& im2, vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2, Mat& dsp1, Mat& dsp2 )
{
    fd->detect ( im1, kpts1 );
    fd->detect ( im2, kpts2 );

    // unique_keypoint ( kpts1 );
    // unique_keypoint ( kpts2 );

    de->compute ( im1, kpts1, dsp1 );
    de->compute ( im2, kpts2, dsp2 );
}

void match_with_NORM_HAMMING(const Ptr<DescriptorMatcher>& matcher, const Mat& des1, const Mat& des2, vector<DMatch>& matches, double threshold_dis)
{
    vector<DMatch> all_matches;
    matcher->match ( des1, des2, all_matches );

    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < all_matches.size(); i++ ) {
        double dist = all_matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    cout << "-- Max dist:" << max_dist;
    cout << "-- Min dist:" << min_dist << endl;

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < all_matches.size (); i++ ) {
        if ( all_matches[i].distance <= max ( 2 * min_dist, threshold_dis ) ) {
            matches.push_back ( all_matches[i] );
        }
    }

    cout << "NORM_HAMMING match:" << all_matches.size () << " -> " << matches.size () << endl;
}


void match_with_knnMatch( const Ptr<DescriptorMatcher>& matcher, const Mat& des1, const Mat& des2,
                          vector<DMatch>& matches, float minRatio)
{
    const int k = 2;

    vector<vector<DMatch>> knnMatches;
    matcher->knnMatch ( des1, des2, knnMatches, k );

    for ( size_t i = 0; i < knnMatches.size (); i++ ) {
        const DMatch& bestMatch = knnMatches[i][0];
        const DMatch& betterMatch = knnMatches[i][1];

        float  distanceRatio = bestMatch.distance / betterMatch.distance;
        if ( distanceRatio < minRatio )
            matches.push_back ( bestMatch );
    }

    cout << "knnMatches:" << knnMatches.size () << " -> " << matches.size () << endl;
}

Ptr<DescriptorMatcher> getMatchTypeNormal ( const int normType )
{
    if (normType == NORM_L2 || normType == NORM_HAMMING) {
        return new BFMatcher ( normType, true );

    }

    cout << "Seems that the normType is not NORM_L2 or NORM_HAMMING, use the default NORM_L2" << endl;
    return new BFMatcher ( NORM_L2, true );

}


/**
 * \brief
 * [【计算机视觉】OpenCV的最近邻开源库FLANN](http://www.jianshu.com/p/d70d9c8b2bec)
 * The values are taken from http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
 * \param normType
 */
Ptr<DescriptorMatcher> getMatchTypeFlann( const int normType)
{

    Ptr<flann::SearchParams> sp = new flann::SearchParams ( 50 );
    if (normType == NORM_HAMMING) { // ORB
        cout << "Using FlannBased LshIndexed method (HAMMING, like ORB)" << endl;
        Ptr<flann::LshIndexParams> lsh = new flann::LshIndexParams ( 6, 12, 1 );
        // Ptr<flann::LshIndexParams> lsh = new flann::LshIndexParams ( 12, 20, 2 );
        return new FlannBasedMatcher ( lsh, sp );
    }

    cout << "Using FlannBased KDTreebase method (L2, like SIFT SURF)" << endl;
    Ptr<flann::KDTreeIndexParams> kdr = new flann::KDTreeIndexParams ( 5 );

    //Ptr<flann::AutotunedIndexParams> autotune = new flann::AutotunedIndexParams ();

    return new FlannBasedMatcher ( kdr, sp );
}





void kp2pts ( const std::vector<KeyPoint>& keypoints_1,
              const std::vector<KeyPoint>& keypoints_2,
              const std::vector< DMatch >& matches,
              vector<Point2f>& points1,
              vector<Point2f>& points2
            )
{

    points1.clear ();
    points2.clear ();
    points1.reserve ( matches.size () );
    points2.reserve ( matches.size () );

    //-- 把匹配点转换为vector<Point2f>的形式

    for ( auto m : matches ) {
        points1.push_back ( keypoints_1[m.queryIdx].pt );
        points2.push_back ( keypoints_2[m.trainIdx].pt );
    }


}


void print_pts ( vector<Point2f>& points1,
                 vector<Point2f>& points2,
                 int start,
                 int end)
{
    assert ( start >= 0 && start < end && end <= points1.size () );

    //cout << "All points:" << points1.size () << ", points << {" << start << "}-{" << end << "}" << endl;

    printf ( "All points:{%llu}, Key points {%d}-{%d}:\n", points1.size (), start, end );

    for ( int i = start; i < end; i++ ) {
        Point2d p1 = points1[i];
        Point2d p2 = points2[i];
        cout << "i:" << i;
        cout << ",p1:" << p1.x << " " << p1.y ;
        cout << ",p2:" << p2.x << " " << p2.y << endl;
    }


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

void pixel2cam ( const double px, const double py, const Mat& K, double& cx, double& cy )
{
    cx = (px - K.at<double> ( 0, 2 )) / K.at<double> ( 0, 0 );
    cy = (py - K.at<double> ( 1, 2 )) / K.at<double> ( 1, 1 );
}

