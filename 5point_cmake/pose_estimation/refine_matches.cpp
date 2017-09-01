#include "refine_matches.h"

using namespace std;
using namespace cv;


void refineMatcheswithHomography ( const vector<KeyPoint> kps1,const vector<KeyPoint> kps2, vector<DMatch>& matches, const double reprojectionThreshold )
{
    const int minNumbermatchesAllowed = 8;
    if ( matches.size () < minNumbermatchesAllowed )
        return;

    //Prepare data for findHomography
    vector<Point2f> pts1, pts2;

    kp2pts ( kps1, kps2, matches, pts1, pts2 );


    //find homography matrix and get inliers mask
    //vector<uchar> inliersMask ( matches.size () );
    Mat inliersMask;
    findHomography ( pts1, pts2, CV_RANSAC, reprojectionThreshold, inliersMask );
    cout << "inliersMask, channels:" << inliersMask.channels () << ", type:" << inliersMask.type () << ", size:" << inliersMask.size () << endl;
    vector<DMatch> inliers;
    for ( int i = 0; i < inliersMask.rows; i++ ) {
        if ( inliersMask.at<uchar> ( i, 0 ) ) {
            inliers.push_back ( matches[i] );
        }
    }



    cout << "In refineMatcheswithHomography, matches: " << matches.size () << " -> " << inliers.size () << endl;

    matches = inliers;
}


void refineMatchesWithFundmentalMatrix ( const vector<KeyPoint> kps1, const vector<KeyPoint> kps2, vector<DMatch>& matches)
{

    vector<Point2f> pts1, pts2;

    kp2pts ( kps1, kps2, matches, pts1, pts2 );

    //vector<uchar> inliersMask ( matches.size () );
    Mat inliersMask;
    findFundamentalMat ( pts1, pts2, inliersMask, FM_RANSAC );
    cout << "inliersMask, channels:" << inliersMask.channels () << ", type:" << inliersMask.type () << ", size:" << inliersMask.size () << endl;
    vector<DMatch> inliers;
    for ( int i = 0; i < inliersMask.rows; i++ ) {
        if ( inliersMask.at<uchar> ( i, 0 ) ) {
            inliers.push_back ( matches[i] );
        }
    }


    cout << "In refineMatchesWithFundmentalMatrix, matches: " << matches.size () << " -> " << inliers.size () << endl;
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
void unique_keypoint(vector<KeyPoint> &points)
{
    const int kHashDiv = 10000;
    bool hash_list[kHashDiv]= {false};
    size_t kpsize = points.size();
    size_t i,j;
    for(i=0,j=0; i<kpsize; i++) {
        int hash_v = static_cast<int>(points[i].pt.x * points[i].pt.y)%kHashDiv;
        if(!hash_list[hash_v]) {
            hash_list[hash_v]=true;
            points[j]=points[i];
            j++;
        }
    }
    for(i=kpsize-1; i>=j; i--) points.pop_back();
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
    for(int i=0; i<pts_num; i++) {
        int flag = static_cast<int>(mask.at<uchar>(i));
        if(flag) {
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
