It is not so hard to use the APIs of feature detector and descriptors, but not so trivial to find the right/possible usage of them. If the descriptors are binary, it is better to use HAMMING distance when do matching, and the nice combination of different detectors and descriptors is still unknown to me.

### [OpenCV 3: List of available FeatureDetector::create() and DescriptorExtractor::create() options?](https://stackoverflow.com/questions/36691050/opencv-3-list-of-available-featuredetectorcreate-and-descriptorextractorc)

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

``` cpp
// To know which norm type to use: OpenCV 3.x
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


## [OpenCV2:特征匹配及其优化](http://www.cnblogs.com/wangguchangqing/p/4333873.html)

It have some refine with F and H.