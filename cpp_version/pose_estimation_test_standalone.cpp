/*
 * This is our code to evaluate the pose_estimation, with real images.
 *
 * The refine part is worth evaluating, whether they are necessary or not.
 */


#include "pose_estimation_header.h"
#include "getRTAlgo.h"

#include "cmdLine/cmdLine.h"
#include "json.hpp"
#include <fstream>

using namespace std;
using namespace cv;

using json = nlohmann::json;

vector<double> getVector ( const Mat &_t1f )
{
    Mat t1f;
    _t1f.convertTo ( t1f, CV_64F );
    return (vector<double>)(t1f.reshape ( 1, 1 ));
}

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc,char **argv)
{

    string camera_K_file = "H:/projects/SLAM/dataset/K.txt";
    string im1 = "1.jpg";
    string im2 = "2.jpg";
    string output_json_file = "output.json";

    CmdLine cmd;
    cmd.add ( make_option ( 'K', camera_K_file, "camera_K_file" ) );
    cmd.add ( make_option ( 'a', im1, "image1" ) );
    cmd.add ( make_option ( 'b', im2, "image2" ) );
    cmd.add ( make_option ( 'o', output_json_file, "output_json file" ) );


    try {
        if ( argc == 1 ) throw std::string ( "Invalid command line parameter." );
        cmd.process ( argc, argv );
    }
    catch ( const std::string& s ) {
        std::cerr << "Usage: " << argv[0] << ' '
            << "[-K|--camera_K_file - the file stores K values]\n"
            << "[-a|--image1 - the file name of image1, absolute path, eg. H:/dataset/1.jpg]\n"
            << "[-b|--image2 - the name of image2]\n"
            << "[-o|--output_json_file - json file for the R,t]\n"
            << std::endl;

        std::cerr << s << std::endl;
        return EXIT_FAILURE;
    }




    Mat K = (Mat_<double> ( 3, 3 ) << 320, 0, 320, 0, 320, 240, 0, 0, 1);


    //string featureName = "BRISK";
    //string featureName = "BRIEF";
    string featureName = "ORB";
    // string featureName = "SIFT";


    string imgname1 = im1;
    string imgname2 = im2;

    vector<KeyPoint> kpts1;
    vector<KeyPoint> kpts2;
    vector <DMatch> matches;
    //double max_ratio = 0.4;    // 0.4 or 0.6
    //double scale = 1.0;   // 1.0

    extractKeyPointsAndMatches ( featureName, imgname1, imgname2, kpts1, kpts2, matches);
    // extractKeyPointsAndMatches (featureName, imgname1, imgname2, kpts1, kpts2, matches, true);

    // - REFINE
    refineMatcheswithHomography ( kpts1, kpts2, matches );
    
    //calcuateRT_test ( kpts1, kpts2, matches, K );
    Mat R, t;
    vector<Point2f> points1, points2;
    kp2pts ( kpts1, kpts2, matches, points1, points2 );
    print_pts ( points1, points2, 0, 10 );
    //calculateRT_CV3 ( points1, points2, K, R, t );

    calculateRT_5points ( points1, points2, K, R, t, 1000, true );

    DEBUG_RT ( R, t );


    // Save R,t to file
    json j;
    j["im1"] = im1;
    j["im2"] = im2;
    
    std::vector<double> rotation_vec ( 9 );
    
    std::vector<double> K_vec ( 9 );
    
    cout << "R:" << endl;
    cout << R << endl;

    j["R"] = getVector(R); // transpose or not?
    j["t"] = getVector(t);
    //j["K"] = K_vec;

    // write prettified JSON to another file
    cout << "Going to save json to " << output_json_file << endl;
    std::ofstream o ( output_json_file );
    o << std::setw ( 4 ) << j << std::endl;
    cout << "Save json done" << endl;


    
    //system ( "pause" );
    return 0;
}
