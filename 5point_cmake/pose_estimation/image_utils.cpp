#include "image_utils.h"

using namespace std;
using namespace cv;

void resize_and_show ( const Mat& im, int target_height, string name )
{
    Mat im2;
    int height = im.rows;
    int width = im.cols;

    if ( height == 0 || width == 0 ) {
        cout << "Seems that the input image is empty" << endl;
        return;
    }


    double ratio = (target_height + 0.0) / height;

    resize ( im, im2, Size ( static_cast<int>(width*ratio), static_cast<int>(height*ratio) ) );

    cout << "New im size:" << im2.size () << endl;

    imshow ( name, im2 );
    waitKey ( 0 );

}