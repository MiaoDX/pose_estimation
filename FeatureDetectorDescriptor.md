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

## GPU accelerated implements

* [CasHash-CUDA, image matching with cascade hashing](https://github.com/cvcore/CasHash_CUDA)

* [CudaSift](https://github.com/Celebrandil/CudaSift)

* [pyboostcvconverter](https://github.com/Algomorph/pyboostcvconverter)

* [numpy-opencv-converter](https://github.com/spillai/numpy-opencv-converter)

* [Using Opencv Cuda functions from python](https://stackoverflow.com/questions/43828944/using-opencv-cuda-functions-from-python)

### [Matching SM architectures (CUDA arch and CUDA gencode) for various NVIDIA cards](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

## Refine/correct the matched points

I do believe the performance of pose estimation really relays on how good the matches are, so the refine part is more than crucial.

* [OpenCV2:特征匹配及其优化](http://www.cnblogs.com/wangguchangqing/p/4333873.html)
    - Use H and F  

* [undistortPoints, findEssentialMat, recoverPose: What is the relation between their arguments?](http://answers.opencv.org/question/65788/undistortpoints-findessentialmat-recoverpose-what-is-the-relation-between-their-arguments/)
    - Use `correctMatches`