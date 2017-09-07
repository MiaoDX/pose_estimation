#pragma once

#include "common_header.h"
#include <math.h>

using namespace std;
using namespace cv;

Mat EulerRadZYX2R ( double Alfa, double Beta, double Gamma );
Mat EulerDegreeZYX2R ( double Alfa, double Beta, double Gamma );
vector<double> GetEulerRadZYX ( const Mat& R );
vector<double> GetEulerDegreeZYX ( const Mat& R );