from PIL import Image, ImageSequence
import numpy as np
from skimage import io
import os
import pandas as pd
import image_processing


# Use ImageJ macro to generate bead frames, then use this code to
# extract bead PSFs and positions


def generate_STORM_data(directory, samples=500, x_dims=(32, 32), y_dims=1,
                        pixel_size=106, normalise_images=True):
    print("Generating STORM data")
    x_all = np.zeros((1, x_dims[0], x_dims[1]))
    y_all = np.zeros(y_dims)

    for i in range(samples):
        # Load image and truth table
        image = io.imread(f'{directory}/{i}.tif')
        truth = pd.read_csv(f'{directory}/{i}.csv', header=0)

        #  x-y position for PSF localisation and z-position
        data = truth[["x [nm]", "y [nm]", "z [nm]"]]

        pixel_x, pixel_y = round(data["x [nm]"] / pixel_size), round(data["y [nm]"] / pixel_size)
        pixel_locations = pd.DataFrame({"x": pixel_x, "y": pixel_y, "z": data["z [nm]"]})

        filtered_emitters = image_processing.filter_points(pixel_locations)
        x, y = image_processing.get_emitter_data(image, filtered_emitters, normalise=normalise_images)

        x_all = np.append(x_all, x, axis=0)
        y_all = np.append(y_all, y)

    # Remove placeholder zeroth elements
    x_train = np.delete(x_all, 0, axis=0)
    y_train = np.delete(y_all, 0)

    # Reshape x_train for direct input into CNN
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

    return x_train, y_train


def scale_zpos(zpos, center=None, scale=None):

    if center is not None and scale is not None:
        return zpos * scale + center

    center = np.mean(zpos)
    scale = np.std(zpos)
    normalised_data = (zpos - center) / scale

    return normalised_data, center, scale
