#include "refine_with_clst_kclusters.h"

using namespace std;
using namespace alglib;

vector<int> clst_cluster(real_2d_array& xy, int sample_num, int cluster_num = 2)
{
    clusterizerstate s;
    ahcreport rep;
    integer_1d_array cidx;
    integer_1d_array cz;

    clusterizercreate ( s );
    clusterizersetpoints ( s, xy, 2 );
    clusterizerrunahc ( s, rep );

    clusterizergetkclusters ( rep, cluster_num, cidx, cz );
    printf ( "In clst_cluster, %s\n", cidx.tostring ().c_str () );

    vector<int> labels;
    for (int i = 0; i < sample_num; i ++) {
        labels.push_back ( cidx[i] );
    }
    return labels;
}

real_2d_array get_real_2d_zyxts(vector<vector<double>>& rt_vec )
{
    assert ( rt_vec[0].size () == 6 );

    vector<double> zyx_t_data;
    for ( auto e : rt_vec ) {
        zyx_t_data.reserve ( zyx_t_data.size () + e.size () );
        zyx_t_data.insert ( zyx_t_data.end (), e.begin (), e.end () );
    }

    real_2d_array real_2d_zyxts;
    real_2d_zyxts.setcontent( rt_vec.size (), 6, zyx_t_data.data ());
    return real_2d_zyxts;
}

vector<vector<double>> get_faked_rt_vec(void)
{
    /*
    >>> a1 = [0, 0, 0, 0, 0, 0]
    >>> a2 = [0, 0, 0, 0, 1, 0]
    >>> a3 = [0, 0, 0, 0, 0, 1]
    >>> a4 = [0, 0, 0, 0, -1, 0]
    >>> a5 = [0, 0, 0, 0, 0, -1]  # above are alike

    >>> a6 = [0, 0, 0, 0, 10, 0]
    >>> a7 = [0, 0, 0, 0, 0, 10]
    >>> a8 = [0, 0, 0, 0, -10, 0]
    >>> a9 = [0, 0, 0, 0, 0, -10]
    */

    vector<double> a1 = { 0, 0, 0, 0, 0, 0 };
    vector<double> a2 = { 0, 0, 0, 0, 1, 0 };
    vector<double> a3 = { 0, 0, 0, 0, 0, 1 };
    vector<double> a4 = { 0, 0, 0, 0, -1, 0 };
    vector<double> a5 = { 0, 0, 0, 0, 0, -1 };

    vector<double> a6 = { 0, 0, 0, 0, 10, 0 };
    vector<double> a7 = { 0, 0, 0, 0, 0, 10 };
    vector<double> a8 = { 0, 0, 0, 0, -10, 0 };
    vector<double> a9 = { 0, 0, 0, 0, 0, -10 };


    vector<vector<double>> zyxs_ts;

    zyxs_ts.push_back ( a1 );
    zyxs_ts.push_back ( a2 );
    zyxs_ts.push_back ( a3 );
    zyxs_ts.push_back ( a4 );
    zyxs_ts.push_back ( a5 );
    zyxs_ts.push_back ( a6 );
    zyxs_ts.push_back ( a7 );
    zyxs_ts.push_back ( a8 );
    zyxs_ts.push_back ( a9 );

    return zyxs_ts;
}



typedef pair<int, vector<int>> PAIR;
bool bigger_first ( const PAIR& lhs, const PAIR& rhs )
{
    return lhs.second.size() > rhs.second.size();
}

/**
 * \brief [C++ STL中Map的按Key排序和按Value排序](http://blog.csdn.net/iicy266/article/details/11906189)
 * \param labels
 * \return
 */
vector<int> find_mode_indexes ( vector<int> labels )
{
    map<int, vector<int>> lables2index;
    for ( int i = 0; i < labels.size (); i++ ) {
        lables2index[labels[i]].push_back ( i );
    }

    //把map中元素转存到vector中
    vector<PAIR> name_score_vec ( lables2index.begin (), lables2index.end () );
    sort ( name_score_vec.begin (), name_score_vec.end (), bigger_first );

    return name_score_vec[0].second;
}


/**
 * \brief http://en.cppreference.com/w/cpp/algorithm/set_intersection
 * \param labels
 * \param last_chosen_index_list
 * \return
 */
vector<int> get_intersection_sample_index(vector<int> index_list, vector<int> last_chosen_index_list)
{
    std::sort ( index_list.begin (), index_list.end () );
    std::sort ( last_chosen_index_list.begin (), last_chosen_index_list.end () );

    std::vector<int> chosen_list;

    std::set_intersection ( index_list.begin (), index_list.end (),
                            last_chosen_index_list.begin (), last_chosen_index_list.end (),
                            std::back_inserter ( chosen_list ) );

    cout << "In get_intersection_sample_index, we got chosen list" << endl;
    for ( int n : chosen_list )
        std::cout << n << ' ';
    cout << endl;

    return chosen_list;
}



vector<int> get_nice_and_constant_sample_index_list( real_2d_array real_2d_zyxts, int sample_num,  double accept_mode_ration = 0.6 )
{
    vector<int> last_chosen_index_list;
    for ( int i = 0; i < sample_num; i++ ) {
        last_chosen_index_list.push_back ( i );
    }


    for ( int cluster_num = 2; cluster_num < 6; cluster_num++ ) {
        cout << "cluster_num:" << cluster_num << endl;
        vector<int> new_labels = clst_cluster ( real_2d_zyxts, sample_num, cluster_num );

        vector<int> mode_index = find_mode_indexes ( new_labels );


        vector<int> chosen_index_list = get_intersection_sample_index (
                                            mode_index, last_chosen_index_list );

        if (static_cast<double>(chosen_index_list.size())/sample_num < accept_mode_ration ) {
            break;
        }

        last_chosen_index_list = chosen_index_list;
    }

    return last_chosen_index_list;
}


vector<double> mean_rt_vec( vector<vector<double>> rt_vec )
{
    vector<double> mean;

    int sample_num = rt_vec.size (), rt_num = rt_vec[0].size();
    assert ( rt_num == 6 );

    for (int col = 0; col < rt_num; col ++) {
        double sum_col = 0.0;
        for (int row = 0; row < sample_num; row ++ ) {
            sum_col += rt_vec[row][col];
        }
        mean.push_back ( sum_col / sample_num );
    }
    return mean;
}


vector<double> get_nice_and_constant_mean_zyxs_ts ( vector<vector<double>> rt_vec, double accept_mode_ration )
{
    real_2d_array real_2d_zyxts = get_real_2d_zyxts ( rt_vec );

    int sample_num = rt_vec.size ();

    vector<int>chosen_sample_index = get_nice_and_constant_sample_index_list ( real_2d_zyxts, sample_num, accept_mode_ration );


    vector<vector<double>> chosen_rt_vec;
    for (int index: chosen_sample_index) {
        chosen_rt_vec.push_back ( rt_vec[index] );
    }

    return mean_rt_vec ( chosen_rt_vec );
}


