// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2016 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.


#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"


#include "openMVG/exif/exif_IO_EasyExif.hpp"
#include "openMVG/geodesy/geodesy.hpp"

// //- Robust estimation - LMeds (since no threshold can be defined)
#include "openMVG/robust_estimation/robust_estimator_LMeds.hpp"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"
#include "third_party/cmdLine/cmdLine.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <stl/split.hpp>

using namespace openMVG;
using namespace openMVG::geodesy;
using namespace openMVG::sfm;

using namespace std;



std::pair<bool, Vec3> checkGPS_faked
(
    const std::string & filename,
    std::map<std::string, Vec3>& faked_pos
)
{
    std::pair<bool, Vec3> val ( false, Vec3::Zero () );


    const std::string filename_idx = stlplus::basename_part ( filename );


    if ( faked_pos.count ( filename_idx ) > 0 )
    {
        val.first = true;
        val.second = faked_pos.at ( filename_idx );
    }
    else
    {
        val.first = false;
    }


    return val;
}


// Parse the database
std::map<std::string, Vec3> get_faked_GPS
(
    const std::string & faked_GPS_filename
) {

    std::map<std::string, Vec3> faked_pos;

    std::ifstream iFilein ( faked_GPS_filename.c_str () );
    if ( stlplus::is_file ( faked_GPS_filename ) && iFilein )
    {
        std::string line;
        while ( iFilein.good () )
        {
            getline ( iFilein, line );
            if ( !line.empty () )
            {

                std::vector<std::string> values;
                stl::split ( line, ';', values );
                if ( values.size () == 4 )
                {
                    faked_pos.insert_or_assign ( values[0], Vec3 ( atof ( values[1].c_str () ), atof ( values[2].c_str () ), atof ( values[3].c_str () ) ) );
                }

            }
        }
    }

    return faked_pos;
}


int main(int argc, char **argv)
{

  std::string
    sSfM_Data_Filename_In,
    sSfM_Data_Filename_Out,
    sFakedGps_Filename;

  CmdLine cmd;  
  cmd.add(make_option('i', sSfM_Data_Filename_In, "input_file"));
  cmd.add(make_option('o', sSfM_Data_Filename_Out, "output_file"));
  cmd.add ( make_option ( 'f', sFakedGps_Filename, "faked_gps_path" ) );

  try
  {
    if (argc == 1) throw std::string("Invalid command line parameter.");
    cmd.process(argc, argv);
  }
  catch (const std::string& s)
  {
    std::cerr
      << "Usage: " << argv[0] << '\n'
      << " GPS registration of a SfM Data scene,\n"
      << "[-i|--input_file] path to the input SfM_Data scene\n"
      << "[-o|--output_file] path to the output SfM_Data scene\n"
      << "[-f|--faked_gps_path], path of file which contains the faked GPS info\n"
      << std::endl;

    std::cerr << s << std::endl;
    return EXIT_FAILURE;
  }

  if (sSfM_Data_Filename_In.empty() || sSfM_Data_Filename_Out.empty())
  {
    std::cerr << "Invalid input or output filename." << std::endl;
    return EXIT_FAILURE;
  }

  // Load input SfM_Data scene
  SfM_Data sfm_data;
  if (!Load(sfm_data, sSfM_Data_Filename_In, ESfM_Data(ALL)))
  {
    std::cerr
      << "\nThe input SfM_Data file \"" << sSfM_Data_Filename_In
      << "\" cannot be read." << std::endl;
    return EXIT_FAILURE;
  }

  Views new_views;
  std::map<std::string, Vec3> faked_pos = get_faked_GPS ( sFakedGps_Filename );

  for (const auto& view_it : sfm_data.GetViews() )
  {

    View* old_view = view_it.second.get ();

    const std::string view_filename =
        stlplus::create_filespec ( sfm_data.s_root_path, old_view->s_Img_path );


    cout << "Going to add view to " << view_filename << "..." << endl;

    /*
     *       sImgPath,
      view_id,
      intrinsic_id,
      pose_id,
      width,
      height
     */
    //ViewPriors v ( old_view->s_Img_path, old_view->id_view, old_view->id_intrinsic, old_view->id_pose, old_view->ui_width, old_view->ui_height );
    ViewPriors v ( old_view->s_Img_path, new_views.size (), new_views.size (), new_views.size (), old_view->ui_width, old_view->ui_height );

    const std::pair<bool, Vec3> gps_info = checkGPS_faked ( view_filename, faked_pos );

    v.id_intrinsic = old_view->id_intrinsic;

    if(gps_info.first == true )
    {
        v.b_use_pose_center_ = true;
        v.pose_center_ = gps_info.second;
    }
    else
    {
        v.b_use_pose_center_ = true;
        v.center_weight_ = Vec3 ( 0, 0, 0 );
        cout << "Seems no valid GPS, continue to others" << endl;
    }

    
    
    // prior weights
    //if ( prior_w_info.first == true )
    //{
    //    v.center_weight_ = prior_w_info.second;
    //}

    
    // Add the view to the sfm_container
    new_views[v.id_view] = std::make_shared<ViewPriors> ( v );
    cout << "Add view done" << endl;
  }

  sfm_data.views = new_views; // we re-assign it

  // Export the SfM_Data scene in the expected format
  if (Save(
        sfm_data,
        sSfM_Data_Filename_Out.c_str(),
        ESfM_Data(ALL)))
  {
    return EXIT_SUCCESS;
  }
}
