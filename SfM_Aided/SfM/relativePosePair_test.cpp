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

#include "json.hpp"
using json = nlohmann::json;

//#define TEST

#ifdef TEST
void test_relative_pose_of_file()
{   
    const string folder_name = "H:/projects/SLAM/pose_estimation/SfM_Aided/sequential_Marx_1_jpg_Csharp/Localization/";
    const string input_file = folder_name + "/sfm_data_expanded.json", reference_im_name = "2.jpg", query_im_name = "query_1.jpg";
    double thetaz=0, thetay=0, thetax=0, x=0, y=0, z=0;
    relative_pose_of_file ( input_file.c_str() , reference_im_name.c_str(), query_im_name.c_str(), thetaz, thetay, thetax, x, y, z );

    //cout << thetaz, thetay, thetax, x, y, z << endl;
    printf ( "thetaz:%f, thetay:%f, thetax:%f, x:%f, y:%f, z:%f", thetaz, thetay, thetax, x, y, z );

}


int main()
{
    test_relative_pose_of_file ();
    system ( "pause" );
}
#else
int main(int argc, char *argv[])
{


  CmdLine cmd;
  std::string sSfM_Data_Filename;
  string reference_im_name = "reference.jpg";
  string query_im_name = "reference.jpg";
  string output_json_file = "output.json";

  cmd.add( make_option('i', sSfM_Data_Filename, "input_file") );
  cmd.add ( make_option ( 'r', reference_im_name, "reference_im_name" ) );
  cmd.add ( make_option ( 'q', query_im_name, "query_im_name" ) );
  cmd.add ( make_option ( 'o', output_json_file, "output_json file" ) );

  

  try {
      if (argc == 1) throw std::string("Invalid command line parameter.");
      cmd.process(argc, argv);
  } catch (const std::string& s) {
      std::cerr << "Usage: " << argv[0] << '\n'
      << "[-i|--input_file] filename, the SfM_Data file to convert\n"
        << "[-r|--reference_im_name] reference_im_name, the name of the image you want to use as reference\n"
        << "[-q|--query_im_name] query_im_name, the name of image you want to use as query\n"
        << "[-o|--output_json_file - json file for the R,t]\n"
        << std::endl;

      std::cerr << s << std::endl;
      return EXIT_FAILURE;
  }


  // Read the input SfM scene
  SfM_Data sfm_data;
  if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(ALL))) {
    std::cerr << std::endl
      << "The input SfM_Data file \""<< sSfM_Data_Filename << "\" cannot be read." << std::endl;
    return EXIT_FAILURE;
  }


  cout << "use " << reference_im_name <<  " as reference" << endl;
  cout << "use " << query_im_name << " as query" << endl;

  std::pair<bool, openMVG::geometry::Pose3> val = relativePoseStr ( sfm_data, query_im_name, reference_im_name );

    if (val.first == false)
    {
        cout << "Seems we can not get pose of image pair, great chances are that the file name is somewhat wrong" << endl;
        return EXIT_FAILURE;
    }

    
    pose2cmd ( val.second );


    json j;
    j["im1"] = reference_im_name;
    j["im2"] = query_im_name;

    //[https://stackoverflow.com/questions/8443102/convert-eigen-matrix-to-c-array](https://stackoverflow.com/a/40271252/7067150)
    std::vector<double> rotation_vec ( 9 );
    Eigen::Map<Eigen::MatrixXd> ( rotation_vec.data (), 3, 3 ) = val.second.rotation ();


    Vec3 translation_Vec = val.second.translation ();
    std::vector<double> translation_vec{ translation_Vec ( 0 ) , translation_Vec ( 1 ), translation_Vec ( 2 ) };


    j["R"] = rotation_vec;
    j["t"] = translation_vec;
    

    // write prettified JSON to another file
    cout << "Going to save json to " << output_json_file << endl;
    std::ofstream o ( output_json_file );
    o << std::setw ( 4 ) << j << std::endl;
    cout << "Save json done" << endl;



  return EXIT_SUCCESS;
}
#endif