Camera pose estimation.

It is one of the most oldest problems in computer vision. Nowadays, there are many successful implements out there. In OpenCV alone, there are `findFundamentalMat` which can use 8 and 7 points method (also with RANSAC or LMEDS), and in opencv 3.x, there is an additional `findEssentialMat` which uses 5-points algorithm (also with RANSAC or LMEDS) which also have `recoverPose` to determine which of the four possible solution is really good.

However, in my experiments, I encountered some problems:

* The E from `findEssentialMat` dose not equals to $K.T * F * K$
* Sometimes it choose the wrong potential answer in `recoverPose`

So, I want to do extra experiments with different implements of 5-points and see how they really behaves. The candidates are:

* OpenCV `findEssentialMat` and `reocverPose`
* Matlab `estimateEssentialMatrix`
* [Nghia Ho., the one we use in our platform now, with the support by our long time usage](http://nghiaho.com/?p=1675)

# Some Notes

## [Notes of Nghia_Ho](Notes_of_Nghia_Ho.md)

## [FeatureDetectorDescriptor](FeatureDetectorDescriptor.md)


# Time of experiment

20170818, the last comment of Nghia Ho is `APRIL 21, 2016 AT 8:11 PM`


***

Good Luck & Have Fun.