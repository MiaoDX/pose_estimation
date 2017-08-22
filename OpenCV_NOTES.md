Some notes of OpenCV

## [Mat, vector<point2f>，Iplimage等等常见类型转换](http://blog.csdn.net/foreverhehe716/article/details/6749175)

Seems not so right.

``` cpp
vector<Point2f> ptsa = Mat_<Point2f>(x1s);
```

## Parameters makes different

* [param1 of findFundamentalMat](http://docs.opencv.org/3.2.0/d9/d0c/group__calib3d.html)

> Parameter used for RANSAC. It is the maximum distance from a point to an epipolar line in pixels, beyond which the point is considered an outlier and is not used for computing the final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the point localization, image resolution, and the image noise.

So, the defaults value of 3.0f can be a little big.

* [crossCheck for BFMatcher](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html)

> crossCheck which is false by default. If it is true, Matcher returns only those matches with value (i,j) such that i-th descriptor in set A has j-th descriptor in set B as the best match and vice-versa. That is, the two features in both sets should match each other. It provides consistant result, and is a good alternative to ratio test proposed by D.Lowe in SIFT paper.

So, it won't hurt to do this check, right?

* ORB

[Feature detector and descriptor for low resolution images](https://stackoverflow.com/questions/24441626/feature-detector-and-descriptor-for-low-resolution-images/43667447#43667447)

The `nfeatures`, the default 500 can be too small for many cases.
`scaleFactor`: Pyramid decimation ratio, maybe we can use larger values for larger images?

``` python
# OpenCV 3.x
orb = cv2.ORB_create(nfeatures=5000)
```


## Format printf

``` vi
size_t x = ...;
ssize_t y = ...;
printf("%zu\n", x);  // prints as unsigned decimal
printf("%zx\n", x);  // prints as hex
printf("%zd\n", y);  // prints as signed decimal
```