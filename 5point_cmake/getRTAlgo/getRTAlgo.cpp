#include "getRTAlgo.h"
#include "5point.h"

using namespace cv;
using namespace std;

/*����Ĳο�ͼƬ��SIFT��Ϣ*/
int height = 1624;
int width = 1224;
int channels = 4;
int image_bytes_count ;
static vector<Point2f> mpts1;
static vector<Point2f> mpts2;



//-----------------------------------------------------------
// �������ƣ�extractMatchFeaturePoints
//     
// ������
//    - featurename��"SIFT", "BRISK", "FAST"
//    - imagename1,imagename2: IMAGE PATH
//	  - dsp1,dsp2: matched points, returned data
//	  - scale: image scale
// ���أ�NONE
//     
// ˵����NONE
//     
//-----------------------------------------------------------



void extractMatchFeaturePoints ( string featurename, string imagename1, string imagename2, vector<Point2f> &pts1, vector<Point2f> &pts2, double max_ratio, double scale  )
{
	pts1.clear();
	pts2.clear();
    // https://stackoverflow.com/questions/313970/how-to-convert-stdstring-to-lower-case
    transform ( featurename.begin (), featurename.end (), featurename.begin (), ::toupper );
	
#ifndef _CV_VERSION_3
	//��ִ����ȡ������������֮ǰ������ִ�иú���~����
	initModule_nonfree ();
	initModule_features2d ();
#endif // ! _CV_VERSION_3

	cout << "Feature name:" << featurename << endl;
	float sc = static_cast<float>(scale);
	cout << "Scale:" << sc << endl;
	
	Mat img1 = imread(string( imagename1 ),0);
	Mat img2 = imread(string( imagename2 ),0);
	
	Size csz(int(img1.cols*sc),int(img1.rows*sc));
	cv::resize(img1,img1,csz);
	cv::resize(img2,img2,csz);

	//detect keypoints
	vector<KeyPoint> kpts1,kpts2;
	Mat dsp1,dsp2;

	Ptr<FeatureDetector> fd;
	Ptr<DescriptorExtractor> de;

#ifdef _CV_VERSION_3

	if ( featurename == "BRISK"  ) {
		fd = de = BRISK::create ( 30, 3, 1.0F ); // These are now the default values
	}
	else if ( featurename == "FAST" ) {
		fd = FastFeatureDetector::create ( 40, true ); // it is not default 10
		de = xfeatures2d::FREAK::create ();
	}
	else {    //SIFT
		fd = de = xfeatures2d::SIFT::create ();
	}
#else
	if ( featurename == "BRISK" ) ) {
		fd = de = new BRISK ( 30, 3, 1.0F ); // These are now the default values
	}
	else if ( featurename == "FAST" ) ) {
		fd = new FastFeatureDetector ( 40, true ); // it is not default 10
		de = new FREAK ();
	}
	else {    //SIFT
		fd = de = new SIFT ();
	}
#endif // _CV_VERSION_3




	fd->detect ( img1, kpts1 );
	fd->detect ( img2, kpts2 );

	unique_keypoint ( kpts1 );
	unique_keypoint ( kpts2 );

	de->compute ( img1, kpts1, dsp1 );
	de->compute ( img2, kpts2, dsp2 );

	// printf("keypoints number: %d %d\n", kpts1.size(), kpts2.size());
	cout << "keypoints number. kpts1.size:" << kpts1.size () << ", kpts2.size:" << kpts2.size () << endl;

	//match keypoints
	Ptr<DescriptorMatcher> dm;
	if( featurename == "SIFT" )
		dm= DescriptorMatcher::create("BruteForce");
	else 
		dm= DescriptorMatcher::create("BruteForce-Hamming");

	vector<vector<DMatch>> mh;
	dm->knnMatch(dsp1,dsp2,mh,2);

	//vector<DMatch> mt;

	for ( int i = 0; i < mh.size (); i++ ) {
		double ratio = mh[i][0].distance / mh[i][1].distance;
		if(ratio < max_ratio ){
			Point2f p1 = kpts1[mh[i][0].queryIdx].pt;
			Point2f p2 = kpts2[mh[i][0].trainIdx].pt;
			p1.x/=sc;
			p1.y/=sc;
			p2.x/=sc;
			p2.y/=sc;
			pts1.push_back(p1);
			pts2.push_back(p2);
		}
	}

	cout << "Matched points number:" << pts1.size () << endl;

}


//-----------------------------------------------------------
// �������ƣ�unique_keypoint
//     
// ������points: ������
// ���أ�NONE
//     
// ˵����ɸѡ��
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
// �������ƣ�matchPointsRansac
//     
// ������pts1, pts2: ƥ���
// ���أ�NONE
//     
// ˵����ɸѡ��
//     
//-----------------------------------------------------------
void matchPointsRansac(vector<Point2f> &pts1,vector<Point2f> &pts2)
{

	//���ݰ˵㷨���㱾�ʾ�����ɸ��
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

	//���ݵ�Ӧ������ɸ��
	Mat hMat = findHomography(pts1,pts2,CV_RANSAC,3.0);

	cout << "Matched points number after RANSAC:" << pts1.size () << endl;
}



bool calculateRT_5points (const vector<Point2f> vpts1, const vector<Point2f> vpts2, double K[9], Mat& R, Mat& t, int ptsLimit, bool showAll)
{
    int matchNum = static_cast<int>(vpts1.size ());
    cout << "matched points number: " << matchNum << endl;

    if ( matchNum < 5 ) return false;
    int maxNum = matchNum < ptsLimit ? matchNum : ptsLimit;

    double *pts1 = static_cast<double *>(malloc ( sizeof ( double ) * maxNum * 2 ));
    double *pts2 = static_cast<double *>(malloc ( sizeof ( double ) * maxNum * 2 ));

    /*
    for ( int i = 0; i < matchNum; i++ ) {
        if ( i < maxNum ) {
            pts1[i * 2] = vpts1[i].x;
            pts1[i * 2 + 1] = vpts1[i].y;
            pts2[i * 2] = vpts2[i].x;
            pts2[i * 2 + 1] = vpts2[i].y;
        }
        else {
            srand ( static_cast<int>(time ( 0 )) );
            int dj = rand () % i;
            if ( dj < maxNum ) {
                pts1[dj * 2] = vpts1[i].x;
                pts1[dj * 2 + 1] = vpts1[i].y;
                pts2[dj * 2] = vpts2[i].x;
                pts2[dj * 2 + 1] = vpts2[i].y;
            }
        }
    }
    */

    for ( int i = 0; i < matchNum; i++ ) {
        pts1[i * 2] = vpts1[i].x;
        pts1[i * 2 + 1] = vpts1[i].y;
        pts2[i * 2] = vpts2[i].x;
        pts2[i * 2 + 1] = vpts2[i].y;
    }

    matchNum = maxNum;
    double invk[9];
    getInvK ( invk, K );


    for ( int i = 0; i < matchNum; i++ ) {
        transformPoint ( invk, pts1[i * 2], pts1[i * 2 + 1] );
        transformPoint ( invk, pts2[i * 2], pts2[i * 2 + 1] );
    }
    vector <cv::Mat> E; // essential matrix
    vector <cv::Mat> P;
    vector<int> inliers;

    bool ret = Solve5PointEssential ( pts1, pts2, matchNum, E, P, inliers ); // ��4����õ�1�����Ž⣻P��ӳ����� [R|t]
    //cout<<ret<<endl;
    free ( pts1 );
    free ( pts2 );
    //pts1=pts2=nullptr;


    cout << "============== Solve5PointEssential =============" << endl;
    size_t best_index = -1;
    if ( ret ) {
        for ( size_t i = 0; i < E.size (); i++ ) {
            if ( cv::determinant ( P[i] ( cv::Range ( 0, 3 ), cv::Range ( 0, 3 ) ) ) < 0 ) P[i] = -P[i];
            if(showAll )
            {
                R = P[i].colRange ( 0, 3 );
                t = P[i].colRange ( 3, 4 );
                DEBUG_RT ( R, t );
            }
            
            if ( best_index == -1 || inliers[best_index] < inliers[i] ) best_index = i;
        }
    }
    else {
        cout << "Could not find a valid essential matrix" << endl;
        return false;
    }
    cout << "============== Solve5PointEssential =============" << endl;
    
    //cout<<"best index:"<<best_index<<endl;
    cv::Mat p_mat = P[best_index];
    cv::Mat Ematrix = E[best_index];

    R = p_mat.colRange ( 0, 3 );
    t = p_mat.colRange ( 3, 4 );

    return true;
}


bool calculateRT_5points (vector<Point2f> vpts1, vector<Point2f> vpts2, double K[9],
	double &rotate_x,double &rotate_y,double &rotate_z, 
	double &move_x,double &move_y,double &move_z, int ptsLimit)
{


    Mat R, t;
    calculateRT_5points ( vpts1, vpts2, K, R, t, ptsLimit, true );

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


void getInvK(double invk[9],double K[9]){
	Mat kmat(3,3,CV_64FC1,K);
	Mat invkmat = kmat.inv();
	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			invk[i*3+j]=invkmat.at<double>(i,j);
		}
	}
}


void transformPoint(double H[9],double &x,double &y){
	double v[3] = {x,y,1};
	double res[3]={0};
	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			res[i] += v[j] * H[i*3+j];
		}
	}
	if(abs(res[2]) > 1e-7){
		x = res[0]/res[2];
		y = res[1]/res[2];
	}
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

    //������������ϵ��������תŷ���ǣ���ת�����ת����������ϵ��
    //��ת˳��Ϊz��y��x

    const double PI = 3.14159265358979323846;
    double thetaz = atan2 ( r21, r11 ) / PI * 180;
    double thetay = atan2 ( -1 * r31, sqrt ( r32*r32 + r33*r33 ) ) / PI * 180;
    double thetax = atan2 ( r32, r33 ) / PI * 180;

    cout << "thetaz:" << thetaz << " thetay:" << thetay << " thetax:" << thetax << endl;
}

void DEBUG_RT ( const Mat& R, const Mat& t )
{
    Mat r;
    cv::Rodrigues ( R, r ); // rΪ��ת������ʽ����Rodrigues��ʽת��Ϊ����

    //cout << "R=" << endl << R << endl;
    //cout << "r=" << endl << r << endl;
    rotate_angle ( R );
    cout << "t:" << t.t() << endl;
}

void calculateRT_CV3 (
    const vector<Point2f> points1,
    const vector<Point2f> points2,
    const Mat K,
    Mat& R, Mat& t )
{
    assert ( points1.size () > 0 && points1.size () == points2.size () && K.size () == Size ( 3, 3 ) );
    R.release (); t.release ();


    //-- �����������
    Mat fundamental_matrix = findFundamentalMat ( points1, points2, CV_RANSAC, 0.1, 0.99 );
    //cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;
    // -- F -> E
    Mat E_f_F;
    essentialFromFundamental ( fundamental_matrix, K, K, E_f_F );
    cout << "E from F:" << endl << E_f_F << endl;
    Mat E_f_F_scaled = scaled_E ( E_f_F );
    cout << "Scaled E:" << endl << E_f_F_scaled << endl;

#ifndef _CV_VERSION_3
    cout << "Seems we are not using OpenCV 3.x, so we can not findEssentialMat, just return" << endl;
    return;
#endif

    //-- ���㱾�ʾ���
    Mat E = findEssentialMat ( points1, points2, K );
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

    //-- ���㵥Ӧ����
    Mat homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
    //cout << "homography_matrix is " << endl << homography_matrix << endl;

    //-- �ӱ��ʾ����лָ���ת��ƽ����Ϣ.
    recoverPose ( E, points1, points2, K, R, t );
}


void essentialFromFundamental ( const Mat &F,
    const Mat &K1,
    const Mat &K2,
    Mat& E )
{
    E = K2.t () * F * K1;
}