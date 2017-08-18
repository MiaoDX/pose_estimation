Camera pose estimation.

It is one of the most oldest problems in computer vision. Nowadays, there are many successful implements out there. In OpenCV alone, there are `findFundamentalMat` which can use 8 and 7 points method (also with RANSAC or LMEDS), and in opencv 3.x, there is an additional `findEssentialMat` which uses 5-points algorithm (also with RANSAC or LMEDS) which also have `recoverPose` to determine which of the four possible solution is really good.

However, in my experiments, I encountered some problems:

* The E from `findEssentialMat` dose not equals to $K.T * F * K$
* Sometimes it choose the wrong potential answer in `recoverPose`

So, I want to do extra experiments with different implements of 5-points and see how they really behaves. The candidates are:

* OpenCV `findEssentialMat` and `reocverPose`
* Matlab `estimateEssentialMatrix`
* [Nghia Ho., the one we use in our platform now, with the support by our long time usage](http://nghiaho.com/?p=1675)



NOTES:

## Questions and answers of Nghia Ho

### Is the output gives the rotational and translation matrix for pts2 (2nd camera position)?

No, the 3×4 projection matrix transforms a 3D point so that it’s relative to the camera’s view. If you want the camera’s position the formula is -R’ * t . Where R is the first 3×3 from the 3×4 matrix, t is the 4th column.

The formulas can be found here [http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html](http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html).

### Does the code calculate P and E from only the corresponding points? Does this mean we can retrieve the calibration matrix K from decomposing P calculated from the 5 point algorithm?

The algorithm assumes you give it **normalized image points**, meaning you are expected to know K before hand already. The P matrix returned does not include K.
REPLY


### I have 2 different cameras with their intrinsic parameters (2 different K matrix). Also I have correspondence points from both the cameras. Now to use this code where should I initialize these intrinsic parameters (K matrix for both cameras)?

Apply them to the points before running the code eg.

ptn1 = K1*pt1
ptn2 = K2*pt2

ptn1, ptn2 becomes input to the code

Correction, K1 should be K1^(-1) (inverse of K1), same for K2.

***

Shall it be like below code..

Point3d u1(pt1.x, pt1.y, 1.0);
Mat_ um1 = K1inv * Mat_(u1);
u1.x = um1(0); u1.y = um1(1); u1.z = um1(2);
ptn1[] = {u1.x/u1.z, u1.y/u1.z};

The 2D point first converted to a 3D point with z value 1 and then after multiply and get back the 2D by x/z and y/z

### More points

Use the RANSAC framework.

1. Randomly pick a set of 5 points
2. Calculate E
3. Apply E to all the points and see how well they “fit” (up to you to define, check Sampson distance)
4. Keep track of which E produces the best fit
5. Repeat until satisfied

You can do one extra step and refine it further by taking all the points that “fit” well and calculate E using all the points (instead of only 5), this will find an average E.

### How to compare the results

The E matrix is determined up to an unknown scale. Divide all the E matrices by E(3,3) to get a better comparison. **I get nearly identical results with Nister’s except the matrix is a transpose of mine (no idea why).**


### Way to get synthetic data

This is my simple code that is supposed to test your implementation.

https://dl.dropboxusercontent.com/u/5257657/n5p0.PNG

psudo:
1) create 5 distinct 3d points
2) create 2 separate camera poses
3) project 3d points from different camera poses (using identity matrix as camera matrix, using 0 distortion) in order to create two arrays of 2d points
4) use these two arrays of 2d points as input to Solve5PointEssential

Result:
vector containing essential matrices is empty
vector containing projection matrices is empty

Quick debugging info:
When populating matrix `b` in 5point.cpp at line 115, cv::determinant(ret_eval) always returns 0.
So, `b` ends up a vector of eleven zeros.


# Time of experiment

20170818, the last comment of Nghia Ho is `APRIL 21, 2016 AT 8:11 PM`
