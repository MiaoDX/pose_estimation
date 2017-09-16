# Utilize SfM for camera pose estimation

The pose estimation get from image pairs will always have one problem -- the magnitude (scale) of translation is unknown. However, since our platform does provide some information of the absolute motion, which we can take use of for much more precious estimation.

The solution is rather simple, we use existing SfM methods to build the 3D module of the scene first, and localize the new image, thus we can get a pretty good estimation.


## PIPELINE

A possible pipeline can be:

```
* SfMInit_ImageListing
* ComputeFeatures
* ComputeMatches
* IncrementalSfM

* Localization one new image
* Get the relative pose between the image and reference one
```

# The implements

As for implements, it will be a tedious work to write all the things from the beginning, and there are some promising open-sourced project, [openMVG](https://github.com/openMVG/openMVG) for example, a library for computer-vision scientists and especially targeted to the Multiple View Geometry community, is pretty good and have a splendid community.

However, I failed to build the latest stable release of it (v1.2, on windows+vs2015), so we choose the develop branch, which is [Fix some issues detected be codacy #1024](https://github.com/openMVG/openMVG/commit/48a6ffeff30a0e5ea78744178758ab170accc283) at present. 

And mostly, we are using the `SfM` and `Localization` module in [software](https://github.com/openMVG/openMVG/tree/develop/src/software) part.

We also add some helper methods:

* Add faked GPS(XYZ) position to our images, providing initial position prior
* Retrieve relative pose with image pair name
* Add GPU SIFT for fast processing (TODO)

# HOW TO

In order to use the program, we should first build the openMVG library. Codes in `SfM` and `Geodesy` is some helper functions we provide for our use, and the original `SfM` and `Localization` module will do most of the tedious work.

To run the codes, which means the [`SfM_Sequential_and_Localization.py`](SfM_Sequential_and_Localization.py) file, we should provide dataset first:

```
dataset_dir
│  fake_gps_file.txt
│
├─sfm_init_data
│      1.jpg
│      2.jpg
│      ...
│      9.jpg
│      reference.jpg
└─sfm_query_data
        add_query_to_im_name.sh
        query_1.jpg
        query_2.jpg
        ...
```


The `sfm_init_data` store images captured the initialize process, which can vary a lot, I provide one possible configuration:

```
8 2 5
7 1 4
9 3 6
```

Number means the order images are taken, for example, `1` and `4` are the first and forth images, and they have known translation provided by our motion equipment. And between the movements, we try not to change the rotation of the platform, so we can get motion prior.


The `fake_gps_file.txt` contains the motion prior we got, and used as `GPS` data for openMVG processing, so we can have direct measure for the translation got from the SfM. The `reference.jpg` is the image we want to regress, and it dose not have any motion prior.

The `sfm_query_data` is the folder contains the query image.

Once the images are provided, it will be pretty straightforward.