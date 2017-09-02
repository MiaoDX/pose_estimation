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



int main ( int argc, char **argv )
{


    vector<int>v{ 1,2,3,-1,-2,-3 };

    double max_value = *max_element ( std::begin ( v ), std::end ( v ) );



    cout << max_value << endl;

    auto result = std::max_element ( v.begin (), v.end () );
    int dis = std::distance ( v.begin (), result );
    std::cout << "max element at: " << dis << '\n';


    system ( "pause" );
    return 0;
}

