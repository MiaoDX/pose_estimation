#ifndef _REFINE_WITH_CLST_KCLUSTERS_H
#define _REFINE_WITH_CLST_KCLUSTERS_H

#include <iostream>
#include <vector>
#include <cassert>
#include <map>
#include <algorithm>
#include <iterator>

#include "dataanalysis.h"

using namespace std;

vector<double> get_nice_and_constant_mean_zyxs_ts ( vector<vector<double>> rt_vec, double accept_mode_ration = 0.6 );

vector<vector<double>> get_faked_rt_vec (void);

#endif
