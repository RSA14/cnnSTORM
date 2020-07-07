from PIL import Image, ImageSequence
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import scipy
import os
import pandas as pd
import image_processing

# Use ImageJ macro to generate bead frames, then use this code to
# extract bead PSFs and positions

X_all = np.zeros((1, 32, 32))
y_all = np.zeros(1)
pixel_size = 106
samples = 500

for i in range(samples):
    # Load image and truth table
    image = io.imread(f'ThunderSTORM/Simulated/LowDensity/{i}.tif')
    truth = pd.read_csv(f'ThunderSTORM/Simulated/LowDensity/{i}.csv', header=0)

    #  x-y position for PSF localisation and z-position to test/train zNN
    data = truth[["x [nm]", "y [nm]", "z [nm]"]]

    pixel_x, pixel_y = round(data["x [nm]"] / pixel_size), round(data["y [nm]"] / pixel_size)
    pixel_locations = pd.DataFrame({"x": pixel_x, "y": pixel_y, "z": data["z [nm]"]})

    filtered_emitters = image_processing.filter_points(pixel_locations)
    X, y = image_processing.get_emitter_data(image, filtered_emitters)

    X_all = np.append(X_all, X, axis=0)
    y_all = np.append(y_all, y)

# Remove placeholder zeroth elements
X_train = np.delete(X_all, 0, axis=0)
y_train = np.delete(y_all, 0)
