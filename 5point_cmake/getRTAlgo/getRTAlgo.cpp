#include "getRTAlgo.h"
#include "5point.h"

using namespace cv;
using namespace std;

/*保存的参考图片的SIFT信息*/
int height = 1624;
int width = 1224;
int channels = 4;
int image_bytes_count ;
static vector<Point2f> mpts1;
static vector<Point2f> mpts2;


//void filterPoints(vector<Point2f> &pts1,vector<Point2f> &pts2,float threshold){
//	if(pts1.size()<4) {
//		//cout<<"match points num<4"<<endl;
//		return;
//	}
//	Mat mask;
//	//Mat hMat = findHomography(pts1,pts2,CV_RANSAC,threshold,mask);
//	int cnt = pts1.size();
//	threshold = 1200.0 / cnt;
//	Mat fMat = findFundamentalMat(pts1,pts2,CV_FM_RANSAC,threshold,0.99,mask);
//
//	vector<Point2f> pts1_,pts2_;
//	int pts_num=pts1.size();
//	for(int i=0;i<pts_num;i++){
//		int flag = (int)mask.at<uchar>(i);
//		if(flag){
//			pts1_.push_back(pts1[i]);
//			pts2_.push_back(pts2[i]);
//		}
//	}
//	pts1=pts1_;
//	pts2=pts2_;
//}
//
//void transformPoint(double H[9],double &x,double &y){
//	double v[3] = {x,y,1};
//	double res[3]={0};
//	for(int i=0;i<3;i++){
//		for(int j=0;j<3;j++){
//			res[i] += v[j] * H[i*3+j];
//		}
//	}
//	if(abs(res[2]) > 1e-7){
//		x = res[0]/res[2];
//		y = res[1]/res[2];
//	}
//}
//
//void unique_keypoint(vector<KeyPoint> &points){
//	const int kHashDiv = 10000;
//	bool hash_list[kHashDiv]={0};
//	int kpsize = points.size();
//	int i,j;
//	for(i=0,j=0;i<kpsize;i++){
//		int hash_v = (int)(points[i].pt.x * points[i].pt.y)%kHashDiv;
//		if(!hash_list[hash_v]){
//			hash_list[hash_v]=true;
//			points[j]=points[i];
//			j++;
//		}
//	}
//	for(i=kpsize-1;i>=j;i--) points.pop_back();
//}
//
//bool getHMatrix(char *featureName,const char *name1,const char *name2,double H[9],double _sc,float MAX_RATIO,vector<Point2f> &pts1,vector<Point2f> &pts2){
//	//read image
//	float sc = _sc;
//	clock_t t1 = clock();
//	cout<<string(name1)<<endl;
//	Mat img1 = imread(string(name1),0);
//	Mat img2 = imread(string(name2),0);
//	
//	//double MAX_RATIO = 0.4;
//	
//	Size csz(int(img1.cols*sc),int(img1.rows*sc));
//	cv::resize(img1,img1,csz);
//	cv::resize(img2,img2,csz);
//
//	ofstream dspLog("dspLog.txt");
//	//detect keypoints
//	vector<KeyPoint> kpts1,kpts2;
//	Mat dsp1,dsp2;
//	clock_t t2;
//	
//	Ptr<FeatureDetector> fd= FeatureDetector::create(featureName);
//	fd->detect(img1,kpts1);
//	fd->detect(img2,kpts2);
//	//cout<<kpts1.size()<<" "<<kpts2.size()<<endl;
//	t2 = clock();
//	Ptr<DescriptorExtractor> de = DescriptorExtractor::create(featureName);
//	unique_keypoint(kpts1);
//	unique_keypoint(kpts2);
//
//	cout<<"特征点找完了kpts1. "<<kpts1.size()<<endl;
//	cout<<"特征点找完了kpts2. "<<kpts2.size()<<endl;
//
//	de->compute(img1,kpts1,dsp1);
//	de->compute(img2,kpts2,dsp2);
//	
//	
//	clock_t t3 = clock();
//	//match keypoints
//	Ptr<DescriptorMatcher> dm;
//	if(0 == strcmp(featureName,"SIFT")) dm= DescriptorMatcher::create("BruteForce");
//	else dm= DescriptorMatcher::create("BruteForce-Hamming");
//	vector<vector<DMatch>> mh;
//	dm->knnMatch(dsp1,dsp2,mh,2);
//	Point2f p1,p2;
//	long long pts_num = mh.size();
//	vector<DMatch> mt;
//	pts1.clear();
//	pts2.clear();
//	float derr = 0;
//	float perr = 0;
//	for(int i=0;i<pts_num;i++){
//		double ratio = mh[i][0].distance / mh[i][1].distance;
//		if(ratio < MAX_RATIO ){
//			////cout<<mh[i][0].distance<<" ";
//			mt.push_back(mh[i][0]);
//			p1=kpts1[mh[i][0].queryIdx].pt;
//			p2=kpts2[mh[i][0].trainIdx].pt;
//			p1.x/=sc;
//			p1.y/=sc;
//			p2.x/=sc;			
//			p2.y/=sc;
//			pts1.push_back(p1);
//			pts2.push_back(p2);
//		}
//	}
//	dspLog.close();
//	//cout<<"keypoint error:"<<perr/2.0/pts1.size()<<endl;
//	//cout<<"descriptor error:"<<derr/128.0/pts1.size()<<endl;
//	clock_t t4 = clock();
//	Mat match_img;
//	//drawMatches(img1,kpts2,img2,kpts2,mt,match_img);
//	//imshow("match",match_img);
//	//waitKey(0);
//	printf("keypoints match:%d %lf\n",pts1.size(),1.0*(t4-t3)/1000);
//	if(pts1.size() < 4) return false;
//
//
//		Mat mask;
//	Mat hMat = findHomography(pts1,pts2,CV_RANSAC,2.0,mask);
//	int pnum = pts1.size();
//	int j=0;
//	for(int i=0;i<pnum;i++){
//		if((int)mask.at<uchar>(i) == 1){
//			pts1[j] = pts1[i];
//			pts2[j] = pts2[i];
//			j++;
//		}
//	}
//	for(j;j<pnum;j++){
//		pts1.pop_back();
//		pts2.pop_back();
//	}
//	cout<<"after ransc:"<<pts1.size()<<endl;
//
//
//	return true;
//}
//
//bool getHMatrix(char *featureName,char *name1,char *name2,double H[9],double _sc){
//	//read image
//	initModule_nonfree();
//	initModule_features2d();
//	printf("%s\n",featureName);
//	const double MAX_RATIO = 0.4;
//	float sc = _sc;
//	clock_t t1 = clock();
//	IplImage * iplImg1 = cvLoadImage(name1,0);
//	IplImage * iplImg2 = cvLoadImage(name2,0);
//	//printf("origin:%d\n",iplImg1->origin);
//	Mat img1(iplImg1,false);
//	Mat img2(iplImg2,false);
//	
//	Size csz(int(iplImg1->width*sc),int(iplImg2->height*sc));
//	cv::resize(img1,img1,csz);
//	cv::resize(img2,img2,csz);
//
//	
//	//detect keypoints
//	vector<KeyPoint> kpts1,kpts2;
//
//	Mat dsp1,dsp2;
//	clock_t t2 = clock();
//	if(0 == strcmp(featureName,"FREAK")){
//		clock_t tt1 = clock();
//		cv::BRISK brk(30,3,1.0F); 
//		brk.create("Feature2D.BRISK");clock_t tt2 = clock();
//		printf("BRISK　create:%d\n",tt2-tt1);
//		clock_t t6 = clock();
//		brk.detect(img1,kpts1);
//		clock_t t7 = clock();
//		brk.detect(img2,kpts2);
//		clock_t t8 = clock();
//		printf("BRISK:%d %d\n",t7-t6,t8-t7);
//		t2 = clock();
//		printf("feature detection:%lf\n",1.0*(t2-t1)/1000);
//		
//		printf("keypoint size:%d\n",kpts1.size());
//		FREAK fe;
//		fe.create("Feature2D.FREAK");
//		clock_t t9 = clock();
//		fe.compute(img1,kpts1,dsp1);
//		clock_t t10 = clock();
//		fe.compute(img2,kpts2,dsp2);
//		clock_t t11 = clock();
//		printf("BRISK extraction:%d %d\n",t11-t10,t10-t9);
//	}else if(0 == strcmp(featureName,"BRISK")){
//		clock_t tt1 = clock();
//		cv::BRISK brk(30,3,1.0F); 
//		brk.create("Feature2D.BRISK");
//		clock_t tt2 = clock();
//		printf("BRISK create:%d\n",tt2-tt1);
//		clock_t tt3 = clock();
//		brk.detect(img1,kpts1);
//		clock_t tt4 = clock();
//		printf("BRISK detect img1:%d\n",tt4-tt3);
//		clock_t tt5 = clock();
//		brk.compute(img1,kpts1,dsp1);
//		clock_t tt6 = clock();
//		printf("BRISK compute img1:%d\n",tt6-tt5);
//		clock_t tt7 = clock();
//		brk.detect(img2,kpts2);
//		clock_t tt8 = clock();
//		printf("BRISK detect img2:%d\n",tt8-tt7);
//		clock_t tt9 = clock();
//		brk.compute(img2,kpts2,dsp2);
//		clock_t tt10 = clock();
//		printf("BRISK compute img2:%d\n",tt10-tt9);
//		
//	}else if(0 == strcmp(featureName,"FAST")){
//		FastFeatureDetector *ft = new FastFeatureDetector(40,true);
//
//		ft->detect(img1,kpts1);
//		ft->detect(img2,kpts2);
//		t2 = clock();
//		printf("feature detection:%d %d %d\n",kpts1.size(),kpts2.size(),t2-t1);
//		
//		FREAK fe;
//		fe.compute(img1,kpts1,dsp1);
//		fe.compute(img2,kpts2,dsp2);
//	}else{
//		Ptr<FeatureDetector> fd= FeatureDetector::create(featureName);	
//		fd->detect(img1,kpts1);
//		fd->detect(img2,kpts2);
//		t2 = clock();
//		printf("feature detection:%lf\n",1.0*(t2-t1)/1000);
//		Ptr<DescriptorExtractor> de = DescriptorExtractor::create(featureName);
//		de->compute(img1,kpts1,dsp1);
//		de->compute(img2,kpts2,dsp2);
//	}
//	clock_t t3 = clock();
//	printf("feature extraction:%d\n",t3-t2);
//	//match keypoints
//	vector<Point2f> pts1,pts2;
//	Ptr<DescriptorMatcher> dm;
//	if(0 == strcmp(featureName,"SIFT")) dm= DescriptorMatcher::create("BruteForce");
//	else dm= DescriptorMatcher::create("BruteForce-Hamming");
//	vector<vector<DMatch>> mh;
//	printf("dsp:%d %d\n",dsp1.rows,dsp2.rows);
//	dm->knnMatch(dsp1,dsp2,mh,2);
//	Point2f p1,p2;
//	long long pts_num = mh.size();
//	vector<DMatch> mt;
//	//cout<<pts_num<<endl;
//	for(int i=0;i<pts_num;i++){
//		double ratio = mh[i][0].distance / mh[i][1].distance;
//		if(ratio < MAX_RATIO ){
//			//mt.push_back(mh[i][0]);
//			////cout<<kpts1.size()<<' '<<mh[i][0].queryIdx<<endl;
//			p1=kpts1[mh[i][0].queryIdx].pt;
//			p2=kpts2[mh[i][0].trainIdx].pt;
//			p1.x/=sc;
//			p1.y/=sc;
//			p2.x/=sc;
//			p2.y/=sc;
//			pts1.push_back(p1);
//			pts2.push_back(p2);
//		}
//	}
//	
//	//Mat match_img;
//	//drawMatches(img1,kpts1,img2,kpts2,mt,match_img);
//	//imshow("match",match_img);
//	//waitKey(0);
//	
//	/*FILE *wf=fopen("indata.txt","a");
//	pts_num=pts1.size();
//	fprintf(wf,"%d\n",pts_num);
//	for(int i=0;i<pts_num;i++) fprintf(wf,"%f %f ",pts1[i].x,pts1[i].y);
//	fprintf(wf,"\n");
//	for(int i=0;i<pts_num;i++) fprintf(wf,"%f %f ",pts2[i].x,pts2[i].y);
//	fprintf(wf,"\n");
//	fclose(wf);
//	*/
//	if(pts1.size() < 4) return false;
//	filterPoints(pts1,pts2,1.0);
//	clock_t t4 = clock();
//	printf("keypoints match:%d %d\n",pts1.size(),t4-t3);
//	//Mat hMat = findHomography(pts1,pts2,0);
//	Mat hMat = findHomography(pts1,pts2,CV_RANSAC,3.0);
//	for(int i=0;i<9;i++){
//		//if(i%3==0) printf("\n");
//		H[i] = hMat.at<double>(i/3,i%3);
//		//printf("%lf ",H[i]);
//	}
//	clock_t t5 = clock();
//	cvReleaseImage(&iplImg1);
//	cvReleaseImage(&iplImg2);
//	return true;
//}
//void getInvK(double invk[9],double K[9]){
//	Mat kmat(3,3,CV_64FC1,K);
//	Mat invkmat = kmat.inv();
//	for(int i=0;i<3;i++){
//		for(int j=0;j<3;j++){
//			invk[i*3+j]=invkmat.at<double>(i,j);
//		}
//	}
//}
//void saveMatch(char *name){
//	if(name == NULL){
//		name = "mpoints.txt";
//	}
//	FILE *fp= fopen(name,"w");
//	int pnum = mpts1.size();
//	fprintf(fp,"%d\n",pnum);
//	for(int i=0;i<pnum;i++){
//		fprintf(fp,"%f %f %f %f\n",mpts1[i].x,mpts1[i].y,mpts2[i].x,mpts2[i].y);
//	}
//	fclose(fp);
//}
//bool getCameraPose_CPU(char *img1,char *img2,double K[9],
//				double &rotate_x,double &rotate_y,double &rotate_z,
//				double &move_x,double &move_y,double &move_z,float scale,int ptsLimit,float ratio){
//	//vector<Point2f> vpts1,vpts2;
//	double H[9];
//	getHMatrix("SIFT",img1,img2,H,(double)scale,ratio,mpts1,mpts2);
//
//	int matchNum = mpts1.size();
//	cout<<"match:"<<matchNum<<endl;
//	if(matchNum < 5) return false;
//	int maxNum = matchNum < ptsLimit ? matchNum : ptsLimit;
//
//	double *pts1 = (double *)malloc(sizeof(double)*maxNum*2);
//	double *pts2 = (double *)malloc(sizeof(double)*maxNum*2);
//
//	for(int i=0;i<matchNum;i++){
//		if(i < maxNum){
//			pts1[i*2] = mpts1[i].x;
//			pts1[i*2+1] = mpts1[i].y;
//			pts2[i*2] = mpts2[i].x;
//			pts2[i*2+1] = mpts2[i].y;
//		}else{
//			srand((int)time(0));
//			int dj = rand()%i;
//			if(dj<maxNum){
//				pts1[dj*2] = mpts1[i].x;
//				pts1[dj*2+1] = mpts1[i].y;
//				pts2[dj*2] = mpts2[i].x;
//				pts2[dj*2+1] = mpts2[i].y;
//			}
//		}
//	}
//	
//	matchNum = maxNum;
//	double invk[9];
//	getInvK(invk,K);
//
//	
//	for(int i=0;i<matchNum;i++){
//		transformPoint(invk,pts1[i*2],pts1[i*2+1]);
//		transformPoint(invk,pts2[i*2],pts2[i*2+1]);
//	}
//	vector <cv::Mat> E; // essential matrix
//    vector <cv::Mat> P;
//	vector<int> inliers;
//	
//	bool ret = Solve5PointEssential(pts1,pts2,matchNum,E,P,inliers);
//	
//	
//
//	//cout<<ret<<endl;
//	free(pts1);
//	free(pts2);
//	pts1=pts2=nullptr;
//
//	int best_index = -1;
//    if(ret) {
//        for(size_t i=0; i < E.size(); i++) {
//            if(cv::determinant(P[i](cv::Range(0,3), cv::Range(0,3))) < 0) P[i] = -P[i];
//			if(best_index == -1 || inliers[best_index] < inliers[i]) best_index = i;
//        }
//    }
//    else {
//        cout << "Could not find a valid essential matrix" << endl;
//		return false;
//    }
//	//cout<<"best index:"<<best_index<<endl;
//	cv::Mat p_mat = P[best_index];
//	cv::Mat Ematrix = E[best_index];
//
//	double rot_x,rot_y,rot_z;
//	rot_y = asin(p_mat.at<double>(2,0));
//	rot_z = asin(-p_mat.at<double>(1,0)/cos(rot_y));
//	rot_x = asin(-p_mat.at<double>(2,1)/cos(rot_y));
//	rotate_x = rot_x*180/CV_PI;
//	rotate_y = rot_y*180/CV_PI;
//	rotate_z = rot_z*180/CV_PI;
//
//	move_x = p_mat.at<double>(0,3);
//	move_y = p_mat.at<double>(1,3);
//	move_z = p_mat.at<double>(2,3);
//	return true;
//}
//
//
//void transmitBitmap(BYTE* data, Mat &grayImg)
//{
//	BYTE *imgData = (BYTE *)malloc(sizeof(BYTE)*width*height*channels);
//	for(int i=0;i<height;i++){
//		BYTE * rowHead = &data[image_bytes_count-(i+1)*width*channels];
//		for(int j=0;j<width*channels;j++){
//			imgData[i*width*channels+j]=rowHead[j];
//		}
//	}
//	Mat img(height,width,CV_MAKETYPE(CV_8U,channels),imgData);
//	imwrite("temp.bmp",img);
//	
//	cvtColor(img,grayImg,COLOR_RGBA2GRAY);
//	free(imgData);  
//}
//
//
//void SaveHMat(char *name1,char *name2,double H[]){
//	string img1(name1);
//	string img2(name2);
//	FindHMat(img1,img2,H);
//}



//-----------------------------------------------------------
// 函数名称：extractMatchFeaturePoints
//     
// 参数：
//    - featurename："SIFT", "BRISK", "FAST"
//    - imagename1,imagename2: IMAGE PATH
//	  - dsp1,dsp2: matched points, returned data
//	  - scale: image scale
// 返回：NONE
//     
// 说明：NONE
//     
//-----------------------------------------------------------
void extractMatchFeaturePoints(char* featureName, char* name1, char* name2, vector<Point2f> &pts1,vector<Point2f> &pts2, double max_ratio, double scale)
{
	pts1.clear();
	pts2.clear();

	
#ifndef _CV_VERSION_3
	//在执行提取特征向量函数之前，必须执行该函数~！！
	initModule_nonfree ();
	initModule_features2d ();
#endif // ! _CV_VERSION_3

	cout << "Feature name:" << featureName << endl;
	float sc = static_cast<float>(scale);
	cout << "Scale:" << sc << endl;
	
	Mat img1 = imread(string(name1),0);
	Mat img2 = imread(string(name2),0);
	
	Size csz(int(img1.cols*sc),int(img1.rows*sc));
	cv::resize(img1,img1,csz);
	cv::resize(img2,img2,csz);

	//detect keypoints
	vector<KeyPoint> kpts1,kpts2;
	Mat dsp1,dsp2;

	Ptr<FeatureDetector> fd;
	Ptr<DescriptorExtractor> de;

#ifdef _CV_VERSION_3
	if ( 0 == strcmp ( featureName, "BRISK" ) ) {
		fd = de = BRISK::create ( 30, 3, 1.0F ); // These are now the default values
	}
	else if ( 0 == strcmp ( featureName, "FAST" ) ) {
		fd = FastFeatureDetector::create ( 40, true ); // it is not default 10
		de = xfeatures2d::FREAK::create ();
	}
	else {    //SIFT
		fd = de = xfeatures2d::SIFT::create ();
	}
#else
	if ( 0 == strcmp ( featureName, "BRISK" ) ) {
		fd = de = new BRISK ( 30, 3, 1.0F ); // These are now the default values
	}
	else if ( 0 == strcmp ( featureName, "FAST" ) ) {
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
	cout << "keypoints number. kpts1.size ():" << kpts1.size () << ", kpts2.size ():" << kpts2.size () << endl;

	//match keypoints
	Ptr<DescriptorMatcher> dm;
	if(0 == strcmp(featureName,"SIFT")) 
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

	cout << "matched points number:" << pts1.size () << endl;

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


bool calculateRT(vector<Point2f> vpts1, vector<Point2f> vpts2, double K[9], 
	double &rotate_x,double &rotate_y,double &rotate_z, 
	double &move_x,double &move_y,double &move_z, int ptsLimit)
{

	int matchNum = static_cast<int>(vpts1.size());
	cout<<"matched points number: "<<matchNum<<endl;
	//ofstream matchLog("match_num.txt");
	//matchLog<<img1<<" "<<img2<<" "<<matchNum<<endl;
	//matchLog.close();
	if(matchNum < 5) return false;
	int maxNum = matchNum < ptsLimit ? matchNum : ptsLimit;

	double *pts1 = static_cast<double *>(malloc(sizeof(double) * maxNum * 2));
	double *pts2 = static_cast<double *>(malloc(sizeof(double) * maxNum * 2));

	for(int i=0;i<matchNum;i++){
		if(i < maxNum){
			pts1[i*2] = vpts1[i].x;
			pts1[i*2+1] = vpts1[i].y;
			pts2[i*2] = vpts2[i].x;
			pts2[i*2+1] = vpts2[i].y;
		}else{
			srand(static_cast<int>(time(0)));
			int dj = rand()%i;
			if(dj<maxNum){
				pts1[dj*2] = vpts1[i].x;
				pts1[dj*2+1] = vpts1[i].y;
				pts2[dj*2] = vpts2[i].x;
				pts2[dj*2+1] = vpts2[i].y;
			}
		}
	}
	
	matchNum = maxNum;
	double invk[9];
	getInvK(invk,K);

	
	for(int i=0;i<matchNum;i++){
		transformPoint(invk,pts1[i*2],pts1[i*2+1]);
		transformPoint(invk,pts2[i*2],pts2[i*2+1]);
	}
	vector <cv::Mat> E; // essential matrix
    vector <cv::Mat> P;
	vector<int> inliers;
	
	bool ret = Solve5PointEssential(pts1,pts2,matchNum,E,P,inliers); // 从4个解得到1个最优解；P：映射矩阵 [R|t]
	
	

	//cout<<ret<<endl;
	free(pts1);
	free(pts2);
	//pts1=pts2=nullptr;

	size_t best_index = -1;
    if(ret) {
        for(size_t i=0; i < E.size(); i++) {
            if(cv::determinant(P[i](cv::Range(0,3), cv::Range(0,3))) < 0) P[i] = -P[i];
			if(best_index == -1 || inliers[best_index] < inliers[i]) best_index = i;
        }
    }
    else {
        cout << "Could not find a valid essential matrix" << endl;
		return false;
    }
	//cout<<"best index:"<<best_index<<endl;
	cv::Mat p_mat = P[best_index];
	cv::Mat Ematrix = E[best_index];

	cout << "Best P:" << endl;
	cout << p_mat << endl;

	cout << "Best Ematrix:" << endl;
	cout << Ematrix << endl;

	Ematrix /= Ematrix.at<double> ( 2, 2 );
	cout << "Scaled E:" << endl << Ematrix << endl;

	Mat R = p_mat.colRange ( 0, 3 );
	Mat t = p_mat.colRange ( 3, 4 );
	Mat r;
	cv::Rodrigues ( R, r ); // r为旋转向量形式，用Rodrigues公式转换为矩阵
	std::cout << "R=" << std::endl << R << std::endl;
	std::cout << "r=" << std::endl << r << std::endl;
	std::cout << "t is " << std::endl << t << std::endl;


	double rot_x,rot_y,rot_z;
	rot_y = asin(p_mat.at<double>(2,0));
	rot_z = asin(-p_mat.at<double>(1,0)/cos(rot_y));
	rot_x = asin(-p_mat.at<double>(2,1)/cos(rot_y));
	rotate_x = rot_x*180/CV_PI;
	rotate_y = rot_y*180/CV_PI;
	rotate_z = rot_z*180/CV_PI;

	move_x = p_mat.at<double>(0,3);
	move_y = p_mat.at<double>(1,3);
	move_z = p_mat.at<double>(2,3);
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