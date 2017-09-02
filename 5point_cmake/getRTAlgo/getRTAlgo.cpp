#include "getRTAlgo.h"
#include "5point.h"
#include <refine_with_clst_kclusters.h>


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


void extractKeyPointsAndMatches( string featurename, const string imagename1, const string imagename2,
                                 vector<KeyPoint>& kpts1, vector<KeyPoint>& kpts2, vector<DMatch>& matches, const bool withFlann )
{
    kpts1.clear();
    kpts2.clear();
    matches.clear();

    // https://stackoverflow.com/questions/313970/how-to-convert-stdstring-to-lower-case
    transform(featurename.begin(), featurename.end(), featurename.begin(), toupper);

    cout << "Feature name:" << featurename << endl;

    Mat img1 = imread(string(imagename1), 0);
    Mat img2 = imread(string(imagename2), 0);

    Ptr<FeatureDetector> fd;
    Ptr<DescriptorExtractor> de;
    getFeatureDetectorDescriptorExtractor(fd, de, featurename);

    Mat dsp1, dsp2;
    extractFeaturesAndDescriptors(fd, de, img1, img2, kpts1, kpts2, dsp1, dsp2);
    cout << "KeyPoints num. kpts1.size:" << kpts1.size() << ", kpts2.size:" << kpts2.size() << endl;


    Ptr<DescriptorMatcher> matcher;
    int normType;
    if (featurename == "ORB" || featurename == "BRIEF" || featurename == "BRISK") {
        // needs HAMMING
        cout << "Going to use NORM_HAMMING, like ORB" << endl;
        normType = NORM_HAMMING;
    }
    else {
        cout << "Going to use NORM_L2, like SIFT" << endl;
        normType = NORM_L2;
    }
    if (withFlann) {
        cout << "Going to use withFlann" << endl;
        matcher = getMatchTypeFlann(normType);
    }
    else {
        matcher = getMatchTypeNormal(normType);
    }

    if (normType == NORM_HAMMING) {
        match_with_NORM_HAMMING(matcher, dsp1, dsp2, matches);
    }
    else {
        match_with_knnMatch(matcher, dsp1, dsp2, matches);
    }


    cout << "Matches num:" << matches.size() << endl;
}






bool _calculateRT_5points_with_ratio_many_results( const vector<Point2f>& vpts1, const vector<Point2f>& vpts2, const Mat& K, vector<Mat>& R_vec, vector<Mat>& t_vec, vector<double>& inliers_ratio_vec, int ptsLimit, bool withDebug )
{
    int npts = static_cast<int>(vpts1.size());
    if (npts < 5) return false;
    int chosenNum = min(npts, ptsLimit);


    cout << "In calculateRT_5points, num of points: " << npts << ", chosenNum:" << chosenNum << endl;

    //pixel2cam
    vector<double> _pts1_cam, _pts2_cam;
    _pts1_cam.resize(chosenNum * 2);
    _pts2_cam.resize(chosenNum * 2);

    for (int i = 0; i < chosenNum; i++) {
        pixel2cam(vpts1[i].x, vpts1[i].y, K, _pts1_cam[i * 2], _pts1_cam[i * 2 + 1]);
        pixel2cam(vpts2[i].x, vpts2[i].y, K, _pts2_cam[i * 2], _pts2_cam[i * 2 + 1]);
    }


    vector<Mat> E; // essential matrix
    vector<Mat> P;
    vector<int> inliers;

    bool ret = Solve5PointEssential(_pts1_cam.data(), _pts2_cam.data(), chosenNum, E, P, inliers); // 从4个解得到1个最优解；P：映射矩阵 [R|t]
    if ( ret == false ) {
        cout << "Could not find a valid essential matrix" << endl;
        return false;
    }

    cout << "============== Solve5PointEssential START =============" << endl;
    printf("Solve5PointEssential() found %llu solutions:\n", E.size());

    for (size_t i = 0; i < E.size(); i++) {
        if (determinant(P[i](Range(0, 3), Range(0, 3))) < 0) P[i] = -P[i];

        Mat R = P[i].colRange(0, 3);
        Mat t = P[i].colRange(3, 4);
        double inliers_ratio = static_cast<double>(inliers[i]) / chosenNum;

        R_vec.push_back(R);
        t_vec.push_back(t);
        inliers_ratio_vec.push_back(inliers_ratio);
    }
    cout << "============== Solve5PointEssential  DONE =============" << endl;

    return true;
}


bool _calculateRT_5points_with_ratio ( const vector<Point2f>& vpts1, const vector<Point2f>& vpts2, const Mat& K, Mat& R, Mat& t, double& inliers_ratio, int ptsLimit, bool withDebug )
{

    vector<Mat> R_vec, t_vec;
    vector<double> inliers_ratio_vec;
    bool ret = _calculateRT_5points_with_ratio_many_results ( vpts1, vpts2, K, R_vec, t_vec, inliers_ratio_vec, vpts1.size (), true );



    if ( ret == false) {
        cout << "Could not find a valid essential matrix" << endl;
        return false;
    }



    auto max_value = std::max_element ( inliers_ratio_vec.begin (), inliers_ratio_vec.end () );
    int best_index = std::distance ( inliers_ratio_vec.begin (), max_value );
    std::cout << "max element at: " << best_index << '\n';

    R = R_vec[best_index];
    t = t_vec[best_index];

    if ( withDebug ) {
        cout << "============== Solve5PointEssential =============" << endl;
    }


    return true;
}


bool calculateRT_5points( const vector<Point2f>& vpts1, const vector<Point2f>& vpts2, const Mat& K, Mat& R, Mat& t, int ptsLimit, bool withDebug )
{
    double inliers_ratio = 0.0;
    return _calculateRT_5points_with_ratio(vpts1, vpts2, K, R, t, inliers_ratio, ptsLimit, withDebug);
}

bool calculateRT_5points( const vector<Point2f>& pts1, const vector<Point2f>& pts2, double K[9],
                          double& rotate_x, double& rotate_y, double& rotate_z,
                          double& move_x, double& move_y, double& move_z, int ptsLimit )
{
    Mat R, t;
    Mat k_M(3, 3, CV_64FC1, K);
    calculateRT_5points(pts1, pts2, k_M, R, t, ptsLimit, true);

    double rot_x, rot_y, rot_z;
    rot_y = asin(R.at<double>(2, 0));
    rot_z = asin(-R.at<double>(1, 0) / cos(rot_y));
    rot_x = asin(-R.at<double>(2, 1) / cos(rot_y));
    rotate_x = rot_x * 180 / CV_PI;
    rotate_y = rot_y * 180 / CV_PI;
    rotate_z = rot_z * 180 / CV_PI;

    move_x = t.at<double>(0, 0);
    move_y = t.at<double>(1, 0);
    move_z = t.at<double>(2, 0);

    return true;
}


void _calculateRT_CV3_with_ratio(
    const vector<Point2f> points1,
    const vector<Point2f> points2,
    const Mat K,
    Mat& R, Mat& t,
    double& inliers_ratio,
    bool withDebug )
{
    assert ( points1.size () > 0 && points1.size () == points2.size () && K.size () == Size ( 3, 3 ) );
    R.release();
    t.release();

#ifndef _CV_VERSION_3
    cout << "Seems we are not using OpenCV 3.x, so no findEssentialMat, just return." << endl;
    return;
#else

    //-- 计算本质矩阵
    Mat E = findEssentialMat(points1, points2, K);

    /*
    if (withDebug) {
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
    */

    //vector<uchar> inliersMask ( points1.size () ); // we can not set as this, since it seems that the Mask is changed shape in the method
    Mat inliersMask;

    //-- 从本质矩阵中恢复旋转和平移信息.re
    recoverPose(E, points1, points2, K, R, t, inliersMask);
    // cout << "inliersMask, channels:" << inliersMask.channels () << ", type:" << inliersMask.type () << ", size:" << inliersMask.size() << endl;
    vector<Point2f> inliers_pts1, inliers_pts2;
    for (int i = 0; i < inliersMask.rows; i++) {
        if (inliersMask.at<uchar>(i, 0)) {
            inliers_pts1.push_back(points1[i]);
            inliers_pts2.push_back(points2[i]);
        }
    }


    if (withDebug) {
        cout << "In recoverPose, points:" << points1.size() << "->" << inliers_pts1.size() << endl;
    }

    inliers_ratio = static_cast<double>(inliers_pts1.size()) / points1.size();
#endif
}

void calculateRT_CV3(
    const vector<Point2f> points1,
    const vector<Point2f> points2,
    const Mat K,
    Mat& R, Mat& t,
    bool withDebug )
{
    double inliers_ratio = 0.0;
    _calculateRT_CV3_with_ratio(points1, points2, K, R, t, inliers_ratio, withDebug);
}


vector<vector<DMatch>> split_matches( const vector<DMatch>& matches, int splitnum = 100 )
{
    vector<vector<DMatch>> split_matches_vec;

    int low_index = 0, high_index = 0;
    do {
        high_index = low_index + splitnum;
        high_index = min(high_index, static_cast<int>(matches.size())); // avoid out of range

        vector<DMatch> tmp;
        tmp.reserve(high_index - low_index);
        copy(matches.begin() + low_index, matches.begin() + high_index, back_inserter(tmp));

        split_matches_vec.push_back(tmp);

        low_index = high_index;
    }
    while (low_index < matches.size());


    return split_matches_vec;
}


vector<vector<double>> split_matches_and_remove_less_confidence(
                        const vector<KeyPoint> kps1, const vector<KeyPoint> kps2, const vector<DMatch>& matches, const Mat& K,
                        int splitnum = 100,
                        double conf_thresh = 0.7 )
{
    vector<vector<double>> zyx_t_vec;

    vector<vector<DMatch>> split_matches_vec = split_matches(matches, splitnum);


    for (auto now_matches: split_matches_vec) {
        vector<Point2f> points1, points2;
        kp2pts(kps1, kps2, now_matches, points1, points2);

        Mat R, t;
        double inliers_ratio = 0.0;
        _calculateRT_CV3_with_ratio(points1, points2, K, R, t, inliers_ratio, true);


        if (inliers_ratio > conf_thresh) {
            vector<double> zyx_t = get_zyx_t_from_R_t(R, t);
            zyx_t_vec.push_back(zyx_t);
        }
    }

    return zyx_t_vec;
}


void calculateRT_CV3_RANSAC( const vector<KeyPoint> kps1, const vector<KeyPoint> kps2, const vector<DMatch>& matches, const Mat& K )
{
    vector<vector<double>> zyx_t_vec = split_matches_and_remove_less_confidence(kps1, kps2, matches, K);

    vector<double> mean_zyx_t = get_nice_and_constant_mean_zyxs_ts(zyx_t_vec);

    cout << "mean_zyx_t:" << endl;
    for (auto e: mean_zyx_t) {
        cout << e << " ";
    }
    cout << endl;
}


vector<vector<double>> split_matches_and_remove_less_confidence_5_points(
                        const vector<KeyPoint> kps1, const vector<KeyPoint> kps2, const vector<DMatch>& matches, const Mat& K,
                        int splitnum = 100,
                        double conf_thresh = 0.7 )
{
    vector<vector<double>> zyx_t_vec;
    vector<vector<DMatch>> split_matches_vec = split_matches(matches, splitnum);

    for (auto now_matches : split_matches_vec) {
        vector<Point2f> points1, points2;
        kp2pts(kps1, kps2, now_matches, points1, points2);

        vector<Mat> R_vec, t_vec;
        vector<double> inliers_ratio_vec;
        _calculateRT_5points_with_ratio_many_results(points1, points2, K, R_vec, t_vec, inliers_ratio_vec, kps1.size(), true);

        for (int i = 0; i < inliers_ratio_vec.size(); i ++) {
            if (inliers_ratio_vec[i] > conf_thresh) {
                vector<double> zyx_t = get_zyx_t_from_R_t(R_vec[i], t_vec[i]);
                zyx_t_vec.push_back(zyx_t);
            }
        }
    }

    return zyx_t_vec;
}


void calculateRT_5points_RANSAC( const vector<KeyPoint> kps1, const vector<KeyPoint> kps2, const vector<DMatch>& matches, const Mat& K )
{
    vector<vector<double>> zyx_t_vec = split_matches_and_remove_less_confidence_5_points(kps1, kps2, matches, K);

    vector<double> mean_zyx_t = get_nice_and_constant_mean_zyxs_ts(zyx_t_vec);

    cout << "mean_zyx_t:" << endl;
    for (auto e : mean_zyx_t) {
        cout << e << " ";
    }
    cout << endl;
}

void calcuateRT_test( const vector<KeyPoint> kps1, const vector<KeyPoint> kps2, const vector<DMatch>& matches, const Mat& K )
{
    Mat R, t;
    vector<Point2f> points1, points2;
    kp2pts(kps1, kps2, matches, points1, points2);
    print_pts(points1, points2, 0, 10);
    calculateRT_CV3(points1, points2, K, R, t);
    DEBUG_RT(R, t);

    Mat R_5, t_5;
    calculateRT_5points(points1, points2, K, R_5, t_5, points1.size());
    DEBUG_RT(R_5, t_5);
}
