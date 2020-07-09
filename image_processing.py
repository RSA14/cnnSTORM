from PIL import Image, ImageSequence
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import scipy
import os
import pandas as pd


def normalise_image(image_data_array, bit_16=True):  # Normalize images to 0 - 1

    if bit_16:
        rescale = 65535
    else:
        rescale = 255

    return np.divide(image_data_array, rescale)


def convert_to_8bit(image):  # Convert from 16-bit to 8-bit
    converted_image = image / 256
    converted_image = converted_image.astype('uint8')
    return converted_image


def cut_image(image, center, width=(16, 16), show=False):
    """

    :param image: numpy array representing image
    :param center: emitter position from truth or STORM
    :param width: window around emitter
    :param show: io.imshow returns cut-out of PSF
    :return: cut out of PSF as a numpy array
    """

    # NOTE: for some reason numpy images seem to have x and y swapped in the logical
    # order as would be for a coordinate point (x,y). I.e. point x,y in the image
    # is actually image[y,x]

    x_width, y_width = width[0], width[1]
    x_min, x_max = int(center[1] - x_width), int(center[1] + x_width)
    y_min, y_max = int(center[0] - y_width), int(center[0] + y_width)

    cut = image[x_min:x_max, y_min:y_max]

    if show:
        io.imshow(cut)

    return cut


def filter_points(points, bound=16):
    """
    Filters emitters using list of emitter locations and a boundary

    :param points: pixel locations of emitters
    :param bound: Exclusion boundary around any emitter
    :return: Viable emitter positions
    """
    rejected_points = set()

    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):

            if abs(points["x"][i] - points["x"][j]) <= bound or abs(points["y"][i] - points["y"][j]) <= bound:
                rejected_points.add(i)
                rejected_points.add(j)

    rejected_points = list(rejected_points)

    return points.drop(rejected_points)


def get_emitter_data(image, points, bound=16, normalise=True):
    """
    Cuts out the (bound, bound) shape around filtered emitters given by points,
    normalise will normalise the final image to 0-1. Can probably combine with filter points

    :param image: Full image of all emitters
    :param points: Filtered list of emitters
    :param bound: Exclusion zone
    :param normalise: Normalise image 0-1
    :return: Array of (bound, bound) PSFs of each emitter and corresponding z-position
    """
    image_data = np.zeros((1, bound, bound))  # Store cut images
    z_data = np.zeros(1)  # Store corresponding z-position

    # Check if emitters are near edge of image so PSF cannot be cropped properly
    for i in points.index:
        if points["x"][i] - bound < 0 or points["x"][i] + bound > image.shape[0]:
            continue
        if points["y"][i] - bound < 0 or points["y"][i] + bound > image.shape[1]:
            continue

        # Cut out image with emitter i at center.
        psf = cut_image(image,
                        (points["x"][i], points["y"][i]),
                        width=(bound, bound))
        if normalise:
            psf = normalise_image(psf)  # Normalise image 0-1 as is common
        z_position = points["z"][i]  # Corresponding z-pos

        # Append to respective arrays
        image_data = np.append(image_data, [psf], axis=0)
        z_data = np.append(z_data, z_position)

    # Delete zeroth initiated elements
    image_data = np.delete(image_data, 0, axis=0)
    z_data = np.delete(z_data, 0)

    return image_data, z_data


