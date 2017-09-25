// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2016 cDc <cdc.seacave@gmail.com>, Pierre MOULON

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "relativePosePair.h"


using namespace openMVG;
using namespace openMVG::sfm;

using namespace cv;
using namespace Eigen;


openMVG::geometry::Pose3 relativePoseIdx(
  const SfM_Data & sfm_data,
    const int query_id,
    const int reference_id
  )
{  
  Views views = sfm_data.GetViews ();

  View* query_view = views.at ( query_id ).get ();
  View* reference_view = views.at ( reference_id ).get ();
  const openMVG::geometry::Pose3 query_pose ( sfm_data.GetPoseOrDie ( query_view ) );
  const openMVG::geometry::Pose3 reference_pose ( sfm_data.GetPoseOrDie ( reference_view ) );

  const openMVG::geometry::Pose3 relative_pose = query_pose * (reference_pose.inverse ());
  

  return relative_pose;  
}



int view_id_from_im_name( const SfM_Data & sfm_data, const string im_name)
{

    for ( Views::const_iterator iter = sfm_data.GetViews ().begin ();
        iter != sfm_data.GetViews ().end (); ++iter )
    {
        const View * view = iter->second.get ();
        if ( stlplus::filename_part ( view->s_Img_path ) == im_name )
        {
            return view->id_view;
        }
    }

    return -1;
}


std::pair<bool, openMVG::geometry::Pose3> relativePoseStr (
    const SfM_Data & sfm_data,
    const string query_im_name,
    const string reference_im_name
)
{
    std::pair<bool, openMVG::geometry::Pose3> val ( false, openMVG::geometry::Pose3() );
    int query_id = view_id_from_im_name ( sfm_data, query_im_name );
    int reference_id = view_id_from_im_name ( sfm_data, reference_im_name );
    

    if(query_id == -1 || reference_id == -1 )
    {
        return  val;
    }

    val.first = true;
    val.second = relativePoseIdx ( sfm_data, query_id, reference_id );
    
    return val;
}






vector<double> pose2cmd(const openMVG::geometry::Pose3& relative_pose )
{
    Eigen::MatrixXd R_mvg = relative_pose.rotation ();
    Eigen::MatrixXd t_mvg = relative_pose.translation ();

    cv::Mat R, t;
    cv::eigen2cv ( R_mvg, R );
    cv::eigen2cv ( t_mvg, t );

    std::vector<double> ra = rotate_angle ( R );
    cout << "thetaz:" << ra[0] << " thetay:" << ra[1] << " thetax:" << ra[2] << endl;

    cout << t.t () << endl;

    vector<double> zyx_t{ ra[0], ra[1], ra[2], t.at<double> ( 0 ),t.at<double> ( 1 ) ,t.at<double> ( 2 ) };
    return zyx_t;
}

bool relative_pose_of_file( const char* input_file, const char* reference_im_name, const char* query_im_name, double& thetaz, double& thetay, double& thetax, double& x, double& y, double& z)
{
    thetaz = 0, thetay = 0, thetax = 0, x = 0, y = 0, z = 0;
 
    if ( !stlplus::file_exists ( input_file ) || std::strlen(reference_im_name) == 0 || std::strlen(query_im_name) == 0 )
    {
        cout << "Invalid file name" << endl;
        return false;
    }

    // Read the input SfM scene
    SfM_Data sfm_data;
    if ( !Load ( sfm_data, input_file, ESfM_Data ( ALL ) ) ) {
        std::cerr << std::endl
            << "The input SfM_Data file \"" << input_file << "\" cannot be read." << std::endl;
        return false;
    }


    cout << "use " << reference_im_name << " as reference" << endl;
    cout << "use " << query_im_name << " as query" << endl;

    std::pair<bool, openMVG::geometry::Pose3> val = relativePoseStr ( sfm_data, static_cast<string>(query_im_name), static_cast<string>(reference_im_name) );

    if ( val.first == false )
    {
        cout << "Seems we can not get pose of image pair, great chances are that the file name is somewhat wrong" << endl;
        return false;
    }


    vector<double> zyx_t = pose2cmd ( val.second );

    thetaz = zyx_t[0], thetay = zyx_t[1], thetax = zyx_t[2], x = zyx_t[3], y = zyx_t[4], z = zyx_t[5];
    
    return true;
}