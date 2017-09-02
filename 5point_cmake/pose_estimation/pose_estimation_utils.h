#pragma once

#include "common_header.h"

using namespace std;
using namespace cv;


Mat scaled_E ( const Mat& E );

vector<double> rotate_angle ( const Mat& R );

void DEBUG_RT ( const Mat& R, const Mat& t );

vector<double> get_zyx_t_from_R_t ( const Mat&R, const Mat&t );



// [Copy from SFM](https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/fundamental.cpp)
void essentialFromFundamental ( const Mat &F,
                                const Mat &K1,
                                const Mat &K2,
                                Mat& E );