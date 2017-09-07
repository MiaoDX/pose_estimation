// #include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "dataanalysis.h"

using namespace alglib;


int main(int argc, char **argv)
{
    //
    // We have a set of points in 2D space:
    //     (P0,P1,P2,P3,P4) = ((1,1),(1,2),(4,1),(2,3),(4,1.5))
    //
    //  |
    //  |     P3
    //  |
    //  | P1
    //  |             P4
    //  | P0          P2
    //  |-------------------------
    //
    // We perform Agglomerative Hierarchic Clusterization (AHC) and we want
    // to get top K clusters from clusterization tree for different K.
    //
    clusterizerstate s;
    ahcreport rep;
    real_2d_array xy = "[[1,1],[1,2],[4,1],[2,3],[4,1.5]]";
    integer_1d_array cidx;
    integer_1d_array cz;

    clusterizercreate(s);
    clusterizersetpoints(s, xy, 2);
    clusterizerrunahc(s, rep);

    // with K=5, every points is assigned to its own cluster:
    // C0=P0, C1=P1 and so on...
    clusterizergetkclusters(rep, 5, cidx, cz);
    printf("%s\n", cidx.tostring().c_str()); // EXPECTED: [0,1,2,3,4]

    // with K=1 we have one large cluster C0=[P0,P1,P2,P3,P4,P5]
    clusterizergetkclusters(rep, 1, cidx, cz);
    printf("%s\n", cidx.tostring().c_str()); // EXPECTED: [0,0,0,0,0]

    // with K=3 we have three clusters C0=[P3], C1=[P2,P4], C2=[P0,P1]
    clusterizergetkclusters(rep, 3, cidx, cz);
    printf("%s\n", cidx.tostring().c_str()); // EXPECTED: [2,2,1,0,1]

    system ( "pause" );
    return 0;
}
