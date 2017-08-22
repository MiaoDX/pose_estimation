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

void getFeatureDetectorDescriptorExtractor ( Ptr<FeatureDetector>& fd, Ptr<DescriptorExtractor>& de, const string featurename)
{
    fd.release (); de.release ();
    // fd->clear (); de->clear (); // not able to clear

#ifdef _CV_VERSION_3

    if ( featurename == "BRISK" ) {
        fd = de = BRISK::create ( 30, 3, 1.0F ); // These are now the default values
    }
    else if ( featurename == "BRIEF" ) { // http://docs.opencv.org/3.2.0/dc/d7d/tutorial_py_brief.html
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
    else if ( featurename == "SURF" )
    {
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
    else if ( featurename == "BRIEF" ) { // http://docs.opencv.org/3.2.0/dc/d7d/tutorial_py_brief.html
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
    else if ( featurename == "SURF" )
    {
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
    for ( int i = 0; i < all_matches.size(); i++ )
    {
        double dist = all_matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    cout << "-- Max dist:" << max_dist;
    cout << "-- Min dist:" << min_dist << endl;

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < all_matches.size (); i++ )
    {
        if ( all_matches[i].distance <= max ( 2 * min_dist, threshold_dis ) )
        {
            matches.push_back ( all_matches[i] );
        }
    }

    cout << "NORM_HAMMING match:" << all_matches.size () << " -> " << matches.size () << endl;
}


void match_with_knnMatch( const Ptr<DescriptorMatcher>& matcher, const Mat& des1, const Mat& des2, 
    vector<DMatch>& matches , float minRatio)
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
    if (normType == NORM_L2 || normType == NORM_HAMMING)
    {
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
    if (normType == NORM_HAMMING) // ORB
    {
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



void refineMatcheswithHomography ( const vector<KeyPoint> kps1,const vector<KeyPoint> kps2, vector<DMatch>& matches, const double reprojectionThreshold ) {
    const int minNumbermatchesAllowed = 8;
    if ( matches.size () < minNumbermatchesAllowed )
        return;

    //Prepare data for findHomography
    vector<Point2f> pts1, pts2;

    kp2pts ( kps1, kps2, matches, pts1, pts2 );

    //find homography matrix and get inliers mask
    vector<uchar> inliersMask ( matches.size () );
    findHomography ( pts1, pts2, CV_RANSAC, reprojectionThreshold, inliersMask );

    vector<DMatch> inliers;
    for ( size_t i = 0; i < inliersMask.size (); i++ ) {
        if ( inliersMask[i] )
            inliers.push_back ( matches[i] );
    }

    cout << "In refineMatcheswithHomography, matches: " << matches.size () << " -> " << inliers.size () << endl;

    // matches.swap ( inliers );
    matches = inliers;
}


void refineMatchesWithFundmentalMatrix ( const vector<KeyPoint> kps1, const vector<KeyPoint> kps2, vector<DMatch>& matches) {
   
    vector<Point2f> pts1, pts2;

    kp2pts ( kps1, kps2, matches, pts1, pts2 );

    vector<uchar> inliersMask ( matches.size () );

    findFundamentalMat ( pts1, pts2, inliersMask, FM_RANSAC );

    vector<DMatch> inliers;
    for ( size_t i = 0; i < inliersMask.size (); i++ ) {
        if ( inliersMask[i] )
            inliers.push_back ( matches[i] );
    }

    cout << "In refineMatchesWithFundmentalMatrix, matches: " << matches.size () << " -> " << inliers.size () << endl;
    // matches.swap ( inliers );
    matches = inliers;
}


//-----------------------------------------------------------
// 函数名称：unique_keypoint
//     
// 参数：points: 特征点
// 返回：NONE
//     
// 说明：筛选点
//     
//-----------------------------------------------------------
void unique_keypoint(vector<KeyPoint> &points){
	const int kHashDiv = 10000;
	bool hash_list[kHashDiv]={false};
	size_t kpsize = points.size();
	size_t i,j;
	for(i=0,j=0;i<kpsize;i++){
		int hash_v = static_cast<int>(points[i].pt.x * points[i].pt.y)%kHashDiv;
		if(!hash_list[hash_v]){
			hash_list[hash_v]=true;
			points[j]=points[i];
			j++;
		}
	}
	for(i=kpsize-1;i>=j;i--) points.pop_back();
}

//-----------------------------------------------------------
// 函数名称：matchPointsRansac
//     
// 参数：pts1, pts2: 匹配点
// 返回：NONE
//     
// 说明：筛选点
//     
//-----------------------------------------------------------
void matchPointsRansac(vector<Point2f> &pts1,vector<Point2f> &pts2)
{

	//根据八点法计算本质矩阵来筛点
	double threshold = 1.0;
	if(pts1.size()<8) {
		return;
	}
	Mat mask;
	size_t cnt = pts1.size();
	threshold = 1200.0 / cnt;
	Mat fMat = findFundamentalMat(pts1,pts2,CV_FM_RANSAC,threshold,0.99,mask);
	vector<Point2f> pts1_,pts2_;
	size_t pts_num=pts1.size();
	for(int i=0;i<pts_num;i++){
		int flag = static_cast<int>(mask.at<uchar>(i));
		if(flag){
			pts1_.push_back(pts1[i]);
			pts2_.push_back(pts2[i]);
		}
	}
	pts1=pts1_;
	pts2=pts2_;

	//根据单应矩阵来筛点
	Mat hMat = findHomography(pts1,pts2,CV_RANSAC,3.0);

	cout << "Matched points number after RANSAC:" << pts1.size () << endl;
}



bool calculateRT_5points ( const vector<Point2f>& vpts1, const vector<Point2f>& vpts2, const Mat& K, Mat& R, Mat& t, int ptsLimit, bool withDebug )
{
    int npts = static_cast<int>(vpts1.size ());
    if ( npts < 5 ) return false;
    int chosenNum = min ( npts, ptsLimit );

    if (withDebug)
    {
        cout << "In calculateRT_5points, num of points: " << npts << ",chosenNum:" << chosenNum << endl;
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
        printf ( "Solve5PointEssential() found %d solutions:\n", E.size () );
    }
    
    size_t best_index = -1;
    if ( ret ) {
        for ( size_t i = 0; i < E.size (); i++ ) {
            if ( cv::determinant ( P[i] ( cv::Range ( 0, 3 ), cv::Range ( 0, 3 ) ) ) < 0 ) P[i] = -P[i];
            
            if( withDebug )
            {
                R = P[i].colRange ( 0, 3 );
                t = P[i].colRange ( 3, 4 );
                printf ( "%d/%d : %d/%d\t", i, E.size(), inliers[i], npts );
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

void resize_and_show ( const Mat& im, int target_height, string name )
{
    Mat im2;
    int height = im.rows;
    int width = im.cols;

    if ( height == 0 || width == 0 )
    {
        cout << "Seems that the input image is empty" << endl;
        return;
    }


    double ratio = (target_height + 0.0) / height;

    resize ( im, im2, Size ( static_cast<int>(width*ratio), static_cast<int>(height*ratio) ) );

    cout << "New im size:" << im2.size () << endl;

    imshow ( name, im2 );
    waitKey ( 0 );

}


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
    
    //-- 从本质矩阵中恢复旋转和平移信息.
    recoverPose ( E, points1, points2, K, R, t );
#endif
}


void essentialFromFundamental ( const Mat &F,
    const Mat &K1,
    const Mat &K2,
    Mat& E )
{
    E = K2.t () * F * K1;
}

void kp2pts ( const std::vector<KeyPoint>& keypoints_1,
    const std::vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches,
    vector<Point2f>& points1,
    vector<Point2f>& points2
)
{
    
    points1.clear (); points2.clear ();
    points1.reserve ( matches.size () ); points2.reserve ( matches.size () );

    //-- 把匹配点转换为vector<Point2f>的形式

    for ( auto m : matches )
    {
        points1.push_back ( keypoints_1[m.queryIdx].pt );
        points2.push_back ( keypoints_2[m.trainIdx].pt );
    }


}


void print_pts ( vector<Point2f>& points1,
    vector<Point2f>& points2 ,
    int start,
    int end)
{
    assert ( start >= 0 && start < end && end <= points1.size () );
    
    //cout << "All points:" << points1.size () << ", points << {" << start << "}-{" << end << "}" << endl;
    
    printf ( "All points:{%d}, Key points {%d}-{%d}:\n", points1.size (), start, end );

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