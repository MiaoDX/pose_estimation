Notes on Feature Detector and Descriptor.

It is not so hard to use the APIs of feature detector and descriptors, but not so trivial to find the right/possible usage of them. And a nice workflow may be appreciate.

# The pose estimation workflow

* Detect the feature points and descriptors
    - SIFT, SURF / ORB, BRIEF, BRISK and others
* Match with the distance of descriptors
    - NORM_L2, NORM_HAMMING
* Refine the matches
    - correctMatches, with Homography, Fundmental
* Calculate the R,t
    - decomposeEssentialMat/recoverPose, decomposeHomographyMat

## Types of available feature detectors and descriptors

[OpenCV 3: List of available FeatureDetector::create() and DescriptorExtractor::create() options?](https://stackoverflow.com/questions/36691050/opencv-3-list-of-available-featuredetectorcreate-and-descriptorextractorc)

Other solution is to test each feature:

if the call to detect() is ok (no exception thrown) ==> feature detection
if the call to compute() is ok ==> feature extraction
if the call to detectAndCompute() is ok ==> both
or looking directly into the source code.

``` vi
BRISK: detector + descriptor
ORB: detector + descriptor
MSER: detector
FAST: detector
AGAST: detector
GFFT: detector
SimpleBlobDetector: detector
KAZE: detector + descriptor
AKAZE: detector + descriptor
FREAK: descriptor
StarDetector: detector
BriefDescriptorExtractor: descriptor
LUCID: descriptor
LATCH: descriptor
DAISY: descriptor
MSDDetector: detector
SIFT: detector + descriptor
SURF: detector + descriptor
```


The default type of descriptors and distance calculate method (norm):

``` cpp
// To know which norm type to use: OpenCV 3.x, OpenCV 2.x do not have the `NormTypes` enum.
Ptr<Feature2D> akaze = AKAZE::create (); 
std::cout << "AKAZE: " << akaze->descriptorType() << " ; CV_8U=" << CV_8U << std::endl;
std::cout << "AKAZE: " << akaze->defaultNorm() << " ; NORM_HAMMING=" << cv::NORM_HAMMING << std::endl;
```

### [Feature Matching in  OpenCV-Python Tutorials](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html)

`cv2.NORM_L2`(default): good for SIFT, SURF
`cv2.NORM_HAMMING`: ORB, BRIEF, BRISK
`cv2.NORM_HAMMING2`: ORB with `VTA_K == 3 or 4`

There is another *prove* out there: [How Does OpenCV ORB Feature Detector Work?](https://stackoverflow.com/questions/7232651/how-does-opencv-orb-feature-detector-work)

### [No more features2d::create ? (Bug #4009)](http://code.opencv.org/issues/4009)

It is okay to accept if things are so.



## Refine/correct the matched points

I do believe the performance of pose estimation really relays on how good the matches are, so the refine part is more than crucial.

* [OpenCV2:特征匹配及其优化](http://www.cnblogs.com/wangguchangqing/p/4333873.html)
    - Use H and F  

* [undistortPoints, findEssentialMat, recoverPose: What is the relation between their arguments?](http://answers.opencv.org/question/65788/undistortpoints-findessentialmat-recoverpose-what-is-the-relation-between-their-arguments/)
    - Use `correctMatches`


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


## Resources online

### Other methods

* [SIFT and ASIFT --- online demo : try if your images match!](http://www.cmap.polytechnique.fr/~yu/research/ASIFT/demo.html)
    - [ASIFT: An Algorithm for Fully Affine Invariant Comparison](http://www.ipol.im/pub/art/2011/my-asift/)
* [OPENCV ASIFT C++ IMPLEMENTATION](http://www.mattsheckells.com/opencv-asift-c-implementation/)

#### OpenGL

* [Win 10 + VS 2015 如何搭建 OpenGL 环境？](https://www.zhihu.com/question/40665433)

* [从零开始搭建 OpenGL 编程环境（Win 10 + VS 2015 + FreeGLUT + GLEW）](http://lemonc.me/opengl-win10-vs2015-setting.html)

* [opengl-tutorial](http://www.opengl-tutorial.org/)

### GPU accelerated implements

* [CasHash-CUDA, image matching with cascade hashing](https://github.com/cvcore/CasHash_CUDA)

* [CudaSift](https://github.com/Celebrandil/CudaSift)

### With python

* [imageAlignment](https://github.com/kif/imageAlignment)

* [pyboostcvconverter](https://github.com/Algomorph/pyboostcvconverter)

* [numpy-opencv-converter](https://github.com/spillai/numpy-opencv-converter)

* [Using Opencv Cuda functions from python](https://stackoverflow.com/questions/43828944/using-opencv-cuda-functions-from-python)



### [Matching SM architectures (CUDA arch and CUDA gencode) for various NVIDIA cards](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

