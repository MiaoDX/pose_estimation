#pragma once

#include "pose_estimation.h"

using namespace std;
using namespace cv;


void resize_and_show ( const Mat& im, int target_height = 640, string name = "Image" );