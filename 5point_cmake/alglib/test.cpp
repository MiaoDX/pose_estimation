// #include "stdafx.h"
//#include <stdlib.h>
//#include <stdio.h>
//#include <math.h>


#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>


using namespace std;



vector<vector<double>> get_faked_rt_vec()
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




/*
 *
 def get_intersection_label_index(new_labels, last_chosen_index_list):
    from scipy.stats import mode
    lable_mode = mode(new_labels).mode
    mode_indexes = np.where(new_labels == lable_mode)
    chosen_list = np.intersect1d(mode_indexes, last_chosen_index_list)

    return chosen_list
 */

typedef pair<int, int> PAIR;
bool bigger_first ( const PAIR& lhs, const PAIR& rhs )
{
    return lhs.second > rhs.second;
}

int find_mode ( vector<int> labels )
{
    map<int, int> lables2index;

    for ( int i = 0; i < labels.size (); i++ ) {
        lables2index[labels[i]] += 1;
    }

    //把map中元素转存到vector中
    vector<PAIR> name_score_vec ( lables2index.begin (), lables2index.end () );
    sort ( name_score_vec.begin (), name_score_vec.end (), bigger_first );


    for ( auto e : name_score_vec ) {
        //cout << e << endl;
        cout << e.first << ":" << e.second << endl;
    }
    return 0;
}











int main ( int argc, char **argv )
{



    vector<vector<double>> rt_vec = get_faked_rt_vec ();
    vector<int> labels {4, 4, 4, 4, 4, 0,1,2, 3};



    find_mode ( labels );




    system ( "pause" );
    return 0;
}

