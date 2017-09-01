#pragma once

#include "refine_matches.h"

using namespace std;
using namespace cv;


Mat scaled_E ( const Mat& E );

void rotate_angle ( const Mat& R );

void DEBUG_RT ( const Mat& R, const Mat& t );

// [Copy from SFM](https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/fundamental.cpp)
void essentialFromFundamental ( const Mat &F,
    const Mat &K1,
    const Mat &K2,
    Mat& E );