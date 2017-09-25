// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2016 cDc <cdc.seacave@gmail.com>, Pierre MOULON

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <string>
#include "openMVG/sfm/sfm.hpp"
#include "third_party/cmdLine/cmdLine.h"

#include <opencv2/core/eigen.hpp>
#include <Rt_transform.h>

#include "shared_EXPORTS.h"

using namespace openMVG;
using namespace openMVG::sfm;

using namespace cv;
using namespace Eigen;


openMVG::geometry::Pose3 relativePoseIdx(
  const SfM_Data & sfm_data,
    const int query_id,
    const int reference_id = 0
  );

int view_id_from_im_name( const SfM_Data & sfm_data, const string im_name);


SHARED_EXPORT 
std::pair<bool, openMVG::geometry::Pose3> relativePoseStr (
    const SfM_Data & sfm_data,
    const string query_im_name = "reference.jpg",
    const string reference_im_name = "reference.jpg"
);


SHARED_EXPORT 
vector<double> pose2cmd(const openMVG::geometry::Pose3& relative_pose );

extern "C" bool SHARED_EXPORT relative_pose_of_file ( const char* input_file, const char* reference_im_name, const char* query_im_name, double& thetaz, double& thetay, double& thetax, double& x, double& y, double& z );