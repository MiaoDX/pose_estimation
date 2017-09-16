# Utilize SfM for camera pose estimation

The pose estimation get from image pairs will always have one problem -- the magnitude (scale) of translation is unknown. However, since our platform does provide some information of the absolute motion, which we can take use of for much more precious estimation.

The solution is rather simple, we use existing SfM methods to build the 3D module of the scene first, and localize the new image, thus we can get a pretty good estimation.


# The implements

Pruning the official code of develop branch, [Fix some issues detected be codacy #1024](https://github.com/openMVG/openMVG/commit/48a6ffeff30a0e5ea78744178758ab170accc283), to be specific, the `SfM` and `Localization` module in [software](https://github.com/openMVG/openMVG/tree/develop/src/software) part.

And add some new methods:

* Add faked GPS(XYZ) position to our images, providing initial position prior
* Add GPU SIFT for fast processing (TODO)