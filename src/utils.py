# ---------------------------------------------------------
# Python Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import sys
import numpy as np
import matplotlib as mpl
import scipy.misc
mpl.use('TkAgg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)


def binarize(imgs):
    return (np.random.uniform(size=imgs.shape) < imgs).astype(np.float32)


def print_metrics(itr, kargs):
    print("*** Iteration {}  ====> ".format(itr))
    for name, value in kargs.items():
        print("{} : {}, ".format(name, value))
    print("")
    sys.stdout.flush()


def _merge(images, size, resize_ratio=1.):
    h, w = images.shape[1], images.shape[2]
    h_ = int(h * resize_ratio)
    w_ = int(w * resize_ratio)

    img_canvas = np.zeros((h_ * size[0], w_ * size[1]))
    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx / size[1])

        image_resize = scipy.misc.imresize(image, size=(h_, w_), interp='bicubic')
        img_canvas[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_resize

    return img_canvas
