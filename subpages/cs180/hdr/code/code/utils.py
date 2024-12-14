"""
HDR stencil code - utils.py
CS 1290 Computational Photography, Brown U.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def create_sample_matrix(file_names, nr_pixels, nr_exposures):
    """
    Takes random samples from the images to use in solve_g.py
    We need at least N(P-1) > (Zmax - Zmin)
    assuming the maximum (Zmax - Zmin) = 255,

    Args:
        file_names:    exposure stack image filenames
        nr_pixels:     number of pixels per image
        nr_exposures:  number of exposures

    Returns:
        sample_matrix: returns a matrix of size:
                       |nr_samples|x|nr_exposures|x|channels|,
                       where nr_samples depends on the the nr_exposures.
                       sample indices are the same for each exposure. so
                       sample_matrix[0, 0, :] was taken at the same pixel
                       location as sample_matrix[0, 1, :].
    """

    nr_samples = int(np.ceil(255. / (nr_exposures - 1)))
    np.random.seed(1290)
    random_sample_indices = np.random.randint(nr_pixels, size=(nr_samples))

    # allocate resulting matrices
    sample_matrix = np.zeros((nr_samples, nr_exposures, 3), dtype=np.uint8)

    for i in range(nr_exposures):

        # import image and convert BGR -> RGB
        image = cv2.cvtColor(cv2.imread(file_names[i]), cv2.COLOR_BGR2RGB)

        red_temp = image[..., 0].flatten()
        red_temp = red_temp[random_sample_indices]

        green_temp = image[..., 1].flatten()
        green_temp = green_temp[random_sample_indices]

        blue_temp = image[..., 2].flatten()
        blue_temp = blue_temp[random_sample_indices]

        sample_matrix[:, i, 0] = red_temp
        sample_matrix[:, i, 1] = green_temp
        sample_matrix[:, i, 2] = blue_temp

    return sample_matrix


def tm_global_scale(hdr_radiance_map):
    """
    Naive tone map function in which exposure values are linearly
    scaled within the range of 0 to 1, using the min and max of the
    exposures. This function is useful as a baseline for comparing
    with the (better) tone mapping functions you will implement.

    Args:
        hdr_radiance_map (np.array): HDR radiance map of the image
                                     with shape (H, W, 3)
    Returns:
        np.array of image with values in range [0.0, 1.0]
    """

    min_value = hdr_radiance_map.min(axis=(0, 1))
    max_value = hdr_radiance_map.max(axis=(0, 1))

    global_scale = np.zeros_like(hdr_radiance_map)

    global_scale[..., 0] = (hdr_radiance_map[..., 0] - min_value[0])/(max_value[0] - min_value[0])
    global_scale[..., 1] = (hdr_radiance_map[..., 1] - min_value[1])/(max_value[1] - min_value[1])
    global_scale[..., 2] = (hdr_radiance_map[..., 2] - min_value[2])/(max_value[2] - min_value[2])

    return np.clip(global_scale, 0., 1.)


def parse_files(directory):
    """
    parse_files()

    Reads in a directory and parses out the exposure values
    files should be named like: "xxx_yyy.jpg" where
    xxx / yyy is the exposure in seconds.

    Args:
        directory:    path to directory containing images

    Returns:
        file_paths:   np.array of paths to files of exposure images
        exposures:    np.array of exposure values corresponding to
                      each exposure image
        nr_exposures: number of exposure images
    """

    file_list = os.listdir(directory)

    file_names = []
    file_paths = []

    # Collect image files
    for i, file_name in enumerate(file_list):
        if file_name[0] == '.':
            continue
        extension = file_name.split(".")[-1].lower()
        is_img = extension in ["jpeg", "jpg", "png", "tiff", "tif"]
        if is_img:
            file_names.append(file_name)
            file_paths.append(os.path.join(directory, file_name))

    exposures = np.zeros((len(file_names)), dtype=np.float64)

    # Get exposure values from file names
    for i, file_name in enumerate(file_names):
        name_split = file_name.split("_")
        name_split[1] = name_split[1].split(".")[0]

        nominator = float(name_split[0])
        denominator = float(name_split[1])
        exposure = nominator / denominator

        exposures[i] = exposure

    # Sort by exposure
    sorted_idxs = np.argsort(exposures)
    exposures = exposures[sorted_idxs]

    file_paths = np.array(file_paths)
    file_paths = file_paths[sorted_idxs]

    # Then inverse to get descending sort order
    exposures = exposures[::-1]
    file_paths = file_paths[::-1]

    nr_exposures = file_paths.shape[0]

    return file_paths, exposures, nr_exposures


def plot_g(g_red, g_green, g_blue):
    """
    Plots recovered inverse g function: (exposure value) -> (pixel value)

    Args:
        g_red, g_green, g_blue: np.arrays in which the index indicates the
                                pixel value, and array values indicate
                                corresponding exposure values; one array
                                for each channel
    Returns: none
    """

    y = np.arange(256)

    plt.figure(figsize=(10, 7))

    plt.subplot(2, 2, 1)
    plt.plot(g_red, y, 'r-')
    plt.xlabel('log Exposure X')
    plt.ylabel('Pixel Value Z')

    plt.subplot(2, 2, 2)
    plt.plot(g_green, y, 'g-')
    plt.xlabel('log Exposure X')
    plt.ylabel('Pixel Value Z')

    plt.subplot(2, 2, 3)
    plt.plot(g_blue, y, 'b-')
    plt.xlabel('log Exposure X')
    plt.ylabel('Pixel Value Z')

    plt.subplot(2, 2, 4)
    plt.plot(g_red, y, 'r-')
    plt.plot(g_green, y, 'g-')
    plt.plot(g_blue, y, 'b-')
    plt.xlabel('log Exposure X')
    plt.ylabel('Pixel Value Z')

    plt.show()
