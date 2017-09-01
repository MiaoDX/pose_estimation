#include "refine_with_clst_kclusters.h"


using namespace std;
using namespace alglib;


int main ( int argc, char **argv )
{

    vector<vector<double>> rt_vec = get_faked_rt_vec ();

    vector<double> mean_rt_vec = get_nice_and_constant_mean_zyxs_ts ( rt_vec );

    cout << "mean_rt_vec:" << endl;
    for (auto e : mean_rt_vec) {
        cout << e << " ";
    }

    system ( "pause" );
    return 0;
}