#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

"""
Rotation Matrix <-> Euler angle

It is transferred from C++ to python, without any change
"""
"""
https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp

    /**  EulerZYX constructs a Rotation from the Euler ZYX parameters:
     *   -  First rotate around Z with alfa,
     *   - then around the new Y with beta,
     *   - then around new X with gamma.
     *
     *  Closely related to RPY-convention.
     *
     *  Invariants:
     *  	- EulerZYX(alpha,beta,gamma) == EulerZYX(alpha +/- PI, PI-beta, gamma +/- PI)
     *  	- (angle + 2*k*PI)
     **/
    inline static Rotation EulerZYX(double Alfa,double Beta,double Gamma) {
        return RPY(Gamma,Beta,Alfa);
    }

    /**   GetEulerZYX gets the euler ZYX parameters of a rotation :
     *   First rotate around Z with alfa,
     *   then around the new Y with beta, then around
     *   new X with gamma.
     *
     *  Range of the results of GetEulerZYX :
     *  -  -PI <= alfa <= PI
     *  -   -PI <= gamma <= PI
     *  -  -PI/2 <= beta <= PI/2
     *
     *  if beta == PI/2 or beta == -PI/2, multiple solutions for gamma and alpha exist.  The solution where gamma==0
     *  is chosen.
     *
     *
     *  Invariants:
     *  	- EulerZYX(alpha,beta,gamma) == EulerZYX(alpha +/- PI, PI-beta, gamma +/- PI)
     *  	- and also (angle + 2*k*PI)
     *
     *  Closely related to RPY-convention.
     **/
    inline void GetEulerZYX(double& Alfa,double& Beta,double& Gamma) const {
        GetRPY(Gamma,Beta,Alfa);
    }

Rotation Rotation::RPY(double roll,double pitch,double yaw)
    {
        double ca1,cb1,cc1,sa1,sb1,sc1;
        ca1 = cos(yaw); sa1 = sin(yaw);
        cb1 = cos(pitch);sb1 = sin(pitch);
        cc1 = cos(roll);sc1 = sin(roll);
        return Rotation(ca1*cb1,ca1*sb1*sc1 - sa1*cc1,ca1*sb1*cc1 + sa1*sc1,
                   sa1*cb1,sa1*sb1*sc1 + ca1*cc1,sa1*sb1*cc1 - ca1*sc1,
                   -sb1,cb1*sc1,cb1*cc1);
    }

// Gives back a rotation matrix specified with RPY convention
void Rotation::GetRPY(double& roll,double& pitch,double& yaw) const
    {
		double epsilon=1E-12;
		pitch = atan2(-data[6], sqrt( sqr(data[0]) +sqr(data[3]) )  );
        if ( fabs(pitch) > (M_PI/2.0-epsilon) ) {
            yaw = atan2(	-data[1], data[4]);
            roll  = 0.0 ;
        } else {
            roll  = atan2(data[7], data[8]);
            yaw   = atan2(data[3], data[0]);
        }
    }


"""


import numpy as np


def GetRPY(R):
    assert R.shape == (3, 3)
    epsilon = 1e-12
    data = R.flatten()

    from math import atan2, pi, sqrt

    pitch = atan2(-data[6], sqrt(data[0] * data[0] + data[3] * data[3]))
    if abs(pitch) > pi / 2 - epsilon:
        yaw = atan2(-data[1], data[4])
        roll = 0.0
    else:
        roll = atan2(data[7], data[8])
        yaw = atan2(data[3], data[0])

    return np.array((roll, pitch, yaw))


def GetEulerRadZYX(R):
    """
    *  -  -PI <= alfa <= PI
    *  -   -PI <= gamma <= PI
    *  -  -PI/2 <= beta <= PI/2
    :param R:
    :return:
    """
    roll, pitch, yaw = GetRPY(R)
    from math import pi

    Alfa = yaw
    Beta = pitch
    Gamma = roll

    assert -pi <= Alfa <= pi
    assert -pi <= Gamma <= pi
    assert -pi / 2 <= Beta <= pi / 2

    return np.array([Alfa, Beta, Gamma]).reshape(3, 1)


def GetEulerDegreeZYX(R):
    from math import pi
    return GetEulerRadZYX(R) / pi * 180


def RPY2R(roll, pitch, yaw):
    from math import sin, cos
    ca1 = cos(yaw)
    sa1 = sin(yaw)
    cb1 = cos(pitch)
    sb1 = sin(pitch)
    cc1 = cos(roll)
    sc1 = sin(roll)

    R_arr = [
        ca1 * cb1, ca1 * sb1 * sc1 - sa1 * cc1, ca1 * sb1 * cc1 + sa1 * sc1,
        sa1 * cb1, sa1 * sb1 * sc1 + ca1 * cc1, sa1 * sb1 * cc1 - ca1 * sc1,
        -sb1, cb1 * sc1, cb1 * cc1
    ]

    return np.array(R_arr).reshape(3, 3)


def EulerZYXDegree2R(zyx_degree):
    from math import pi

    zyx_rad = zyx_degree / 180 * pi

    return EulerZYXRad2R(zyx_rad)


def EulerZYXRad2R(zyx_rad):
    assert zyx_rad.shape == (3, 1)
    from math import pi
    Alfa = zyx_rad[0]
    Beta = zyx_rad[1]
    Gamma = zyx_rad[2]
    assert -pi <= Alfa <= pi
    assert -pi <= Gamma <= pi
    assert -pi / 2 <= Beta <= pi / 2
    return RPY2R(Gamma, Beta, Alfa)


if __name__ == "__main__":

    R = np.array([0, 0, 1, 0, 1, 0, -1, 0, 0]).reshape(3, 3)

    R = np.array([
        0.99943541, 0.00186892, 0.03354636, -0.00198323, 0.99999234, 0.00337439,
        -0.0335398, -0.00343902, 0.99943147
    ]).reshape(3, 3)

    zyx_rad = GetEulerRadZYX(R)
    print(zyx_rad.T)

    zyx_degree = GetEulerDegreeZYX(R)
    print(zyx_degree.T)

    R2 = EulerZYXRad2R(zyx_rad)
    print(R2)

    R3 = EulerZYXDegree2R(zyx_degree)
    print(R3)
