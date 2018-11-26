"""
The [pose3 class](https://github.com/openMVG/openMVG/blob/master/src/openMVG/geometry/pose3.hpp)

[Rotation registration with android phone euler angles](https://github.com/openMVG/openMVG/issues/551)

[](https://openmvg.readthedocs.io/en/latest/openMVG/cameras/cameras/): R:the rotation of the camera to the world frame
"""

import numpy as np
import math
import transforms3d

class Pose3(object):

    def __init__(self, R=np.eye(3), C=np.zeros(3)):
        assert np.allclose(np.linalg.inv(R), R.T)
        C = C.reshape(3,1)

        self.rotation_ = R
        self.center_ = C

    def rotation(self):
        return self.rotation_

    def center(self):
        return self.center_

    def translation(self):
        return -np.dot(self.rotation_, self.center_)

    def apply_point(self, p):
        p = np.array(p).reshape(3, 1)
        return np.dot(self.rotation(), p - self.center())

    def apply_points(self, pts):
        assert pts.shape[0] == 3
        # pts = np.array(pts).reshape(3, -1)
        return np.dot(self.rotation(), pts - self.center())

    def compose(self, P):
        """
        compose current pose to P, that is, Return = self * P
        :param P:
        :return:
        """
        assert type(P) == Pose3
        return Pose3( np.dot(self.rotation(), P.rotation()),  P.center() +  np.dot(P.rotation().T, self.center()))

    def inverse(self):
        return Pose3(self.rotation().T, -( np.dot( self.rotation(), self.center())))


    """
    OTHER HELPER FUNCTIONS
    """

    def __copy__(self):
        return Pose3(self.rotation(), self.center())

    def copy(self):
        return Pose3(self.rotation(), self.center())

    def toCenter6D(self):
        """
        translate pose to 6D representation: cx, cy, cz, euler_z, euler_y, euler_x
        :return: pose using euler angle and center
        """
        return np.append(self.center().T, rotm2zyxEulDegree (self.rotation()))

    def to6D(self):
        """
        translate pose to 6D representation: tx, ty, tz, euler_z, euler_y, euler_x
        :return: pose using euler angle and translation
        """
        t = self.translation().T.ravel()
        euler = rotm2zyxEulDegree(self.rotation())
        # return np.append(t, euler)
        return np.append(np.array(t), np.array(euler))

    def sameAs(self, P):
        assert np.allclose(self.rotation(), P.rotation()) and np.allclose(self.center(), P.center())

    def debug(self):
        info = "R:{}\neuler_zyx:{}\nC:{}\nt:{}\n".format(self.rotation(),
                                                         rotm2zyxEulDegree(self.rotation()).T, self.center().T, self.translation().T)
        print (info)
        return info

    @classmethod
    def fromRt(cls, R, t):
        assert np.allclose(R.T, np.linalg.inv(R))
        R = np.asarray(R)
        C = -R.T.dot(t)

        return Pose3(R, C)

    @classmethod
    def from6D(cls, relative6D):
        """
        Pose from 6D (tx, ty, tz, roll, yaw, pitch)
        :param relative6D:
        :return:
        """
        relative6D = np.array(relative6D)
        R = zyxEulDegree2Rotm(*relative6D[3:])
        t = relative6D[:3]

        return cls.fromRt(R, t)

    @classmethod
    def fromCenter6D(cls, center6D):
        """
        Pose from 6D (cx, cy, cz, roll, yaw, pitch)
        :param pose:
        :return:
        """
        center6D = np.array(center6D)
        assert len(center6D) == 6
        R = zyxEulDegree2Rotm (*center6D[3:])
        return Pose3(R, center6D[:3])

"""
Rotation transform
"""

def zyxEulDegree2Rotm(z_degree, y_degree, x_degree):
    z_rad = math.radians(z_degree)
    y_rad = math.radians(y_degree)
    x_rad = math.radians(x_degree)

    return transforms3d.euler.euler2mat(z_rad, y_rad, x_rad, 'rzyx')

def rotm2zyxEulDegree(R):
    zyx_rad = transforms3d.euler.mat2euler(R, 'rzyx')
    zyx_degree = list(map(math.degrees, zyx_rad))
    return np.array(zyx_degree)


if __name__ == '__main__':

    # p_w = [0, 0, 0]
    # p_w = [-1, -1, -1]
    p_w = [1, 1, 1]

    x_y_z = [2, 0, 0]
    #z_y_x_degree = [-90, -90, -90]
    # z_y_x_degree = [-90, 90, 0]
    z_y_x_degree = [0, -90, 0]
    P = Pose3().fromCenter6D(x_y_z+z_y_x_degree)
    P.debug()
    p_c = P.apply_point(p_w)
    print(p_c)

    p_ws = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
    p_ws = np.array(p_ws).T
    print(p_ws.shape)
    print(p_ws[:, 0])
    p_cs = P.apply_points(p_ws)
    print(p_cs)