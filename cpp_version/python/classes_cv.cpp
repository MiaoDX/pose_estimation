#include "opencv2/core.hpp"
#include <iostream>
#include <getRTAlgo.h>

using namespace cv;
using namespace std;

#include <sstream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
namespace py = pybind11;

class RelativePose5Point
{
public:

    vector<double> pylist_to_vecdouble ( py::list pylist )
    {
        long l = py::len ( pylist );
        vector<double> vec_d;
        vec_d.reserve ( l );
        for ( long i = 0; i<l; ++i ) {
            double x = py::cast<double> ( pylist[i] );
            vec_d.push_back ( x );
        }
        return vec_d;
    }


    vector<Point2f> pylist_to_vecPoint2f( py::list pylist )
    {
        
        vector<double> vecdouble = pylist_to_vecdouble ( pylist );

        long l = vecdouble.size ();
        vector<Point2f> vec_p;
        vec_p.reserve ( l / 2 );
        for ( long i = 0; i<l/2; ++i ) {                     
            vec_p.push_back ( Point2f ( vecdouble[i * 2], vecdouble[i * 2 + 1] ) );
        }

        return vec_p;
    }


    py::list calcRP( py::list list_points1, py::list list_points2, py::list cameraK_9 )
    {
        //Mat K = (Mat_<double> ( 3, 3 ) << 320, 0, 320, 0, 320, 240, 0, 0, 1);
        vector < double > vec_cam9 = pylist_to_vecdouble ( cameraK_9 );
        Mat K = (Mat_<double> ( 3, 3 ) 
            << 
            vec_cam9[0], vec_cam9[1], vec_cam9[2],
            vec_cam9[3], vec_cam9[4], vec_cam9[5],
            vec_cam9[6], vec_cam9[7], vec_cam9[8]);
        Mat R, t;
        vector<Point2f> vec_p1 = pylist_to_vecPoint2f ( list_points1 );
        vector<Point2f> vec_p2 = pylist_to_vecPoint2f ( list_points2 );
        
        
        calculateRT_5points ( vec_p1, vec_p2, K, R, t, 1000, true );


        auto vec_R = cv_mat_vec ( R );
        auto vec_t = cv_mat_vec ( t );

        cout << "In C++, The matrixes are:" << endl;
        cout << "K:" << endl;
        cout << K << endl;
        cout << "R:" << endl;
        cout << R << endl;
        cout << "t:" << endl;
        cout << t << endl;
        DEBUG_RT ( R, t );
        cout << "DONE" << endl;

        py::list two_pylist;
        two_pylist.append ( vec_R );
        two_pylist.append ( vec_t );

        return two_pylist;
    }


    vector<double> getVector ( const Mat &_t1f )
    {
        Mat t1f;
        _t1f.convertTo ( t1f, CV_64F );
        return (vector<double>)(t1f.reshape ( 1, 1 ));
    }

    py::list cv_mat_vec (const Mat& mat)
    {
        py::list vec_l;

        for(auto e : getVector ( mat ) )
        {
            vec_l.append ( e );
        }

        return vec_l;
    }
};


PYBIND11_MODULE ( RelativePose5Point, m )
{
    py::class_<RelativePose5Point> ( m, "RelativePose5Point" )
        .def ( py::init<> () ) // THIS IS MUST
        .def ( "calcRP", &RelativePose5Point::calcRP, py::arg ( "list_points1" ) = "Points in first image", py::arg ( "list_points2" ) = "Points in second image", py::arg("cameraK_9") = "Camera matrix in 1x9 format" )
    ;
}