import numpy as np


def make_homog(points):
    """ Convert a set of points (dim * n array) to
    homogeneous coordinates. """
    points = points.copy()
    return np.vstack((points,np.ones((1,points.shape[1]))))

def convert_to_homog(points):
    """ Convert a set of points (dim * n array) to
    homogeneous coordinates. """
    points = points.copy()
    return np.divide(points[:-1], points[-1])


def cube_points(c,wid):
    """ Creates a list of points for plotting
    a cube with plot. (the first 5 points are
    the bottom square, some sides repeated). """
    p = []
    #bottom
    p.append([c[0]-wid,c[1]-wid,c[2]-wid])
    p.append([c[0]-wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]-wid,c[2]-wid])
    p.append([c[0]-wid,c[1]-wid,c[2]-wid]) #same as first to close plot

    #top
    p.append([c[0]-wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]-wid,c[2]+wid]) #same as first to close plot

    #vertical sides
    p.append([c[0]-wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]-wid])

    # Center
    # p.append([c[0], c[1], c[2]])

    return np.array(p).T


def cube_points9(c,wid):
    """ Creates a list of points for plotting
    a cube with plot. (the first 5 points are
    the bottom square, some sides repeated). """
    p = []
    #bottom
    p.append([c[0]-wid,c[1]-wid,c[2]-wid])
    p.append([c[0]-wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]-wid,c[2]-wid])

    #top
    p.append([c[0]-wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]+wid])

    # Center
    p.append([c[0], c[1], c[2]])

    return np.array(p).T

if __name__ == '__main__':

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    # mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    center = [0, 0, 20]
    wid = 5
    cube = cube_points(center, wid)

    print(cube)

    # ax.plot(xs=cube[0], ys=cube[1], zs=cube[2], label='parametric curve')
    ax.scatter(xs=cube[0], ys=cube[1], zs=cube[2], label=None)
    ax.legend()


    ax.set_xlim(-20, 20)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-5, 30)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.grid()
    plt.show()


