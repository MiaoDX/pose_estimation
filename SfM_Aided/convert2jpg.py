# import the necessary packages
import cv2
import argparse


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_dir", required=True,
        help="path to input image directory")
    ap.add_argument("-a", "--format_in", required=True,
        help="the input format")
    ap.add_argument("-b", "--format_out", required=True,
        help="the output format")
    args = vars(ap.parse_args())


    image_dir = args["image_dir"]
    format_in = args["format_in"]
    format_out = args["format_out"]

    print("The arguments: image_dir:{}, format_in:{}, format_out:{}".format(image_dir, format_in, format_out))

    import os
    from os import listdir
    from os.path import isfile, join, basename, splitext

    for f in listdir(image_dir):
        f_in = join(image_dir, f)
        print("Deal with file {}".format(f_in))
        f_no_ext, ext_in = splitext(basename(f_in))
        # print("f_no_ext:{}, ext_in:{}".format(f_no_ext, ext_in))
        if isfile(f_in) and ext_in == "."+format_in:
            im = cv2.imread(f_in)
            f_out = join(image_dir, f_no_ext+"."+args["format_out"])
            cv2.imwrite(f_out, im)
            

    print("All done")