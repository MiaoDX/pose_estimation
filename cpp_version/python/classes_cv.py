#!/usr/bin/env python

import numpy as np
import RelativePose5Point

t = RelativePose5Point.RelativePose5Point()

l1 = list(range(10))
l2 = list(range(10, 20))


K = np.array([320, 0, 320, 0, 320, 240, 0, 0, 1]).reshape(3, 3)

# calcRP( py::list list_points1, py::list list_points2, py::list cameraK_9 )

R, t = t.calcRP(l1, l2, list(K.flatten()))
print("R:{}".format(R))
print("t:{}".format(t))