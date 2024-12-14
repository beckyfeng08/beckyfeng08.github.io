"""
HDR stencil code - student.py
CS 1290 Computational Photography, Brown U.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


# ========================================================================
# RADIANCE MAP RECONSTRUCTION
# ========================================================================


def solve_g(Z, B, l, w):
    """
    Given a set of pixel values observed for several pixels in several
    images with different exposure times, this function returns the
    imaging system's response function g as well as the log film irradiance
    values for the observed pixels.

    Args:
        Z[i,j]: the pixel values of pixel location number i in image j.
        B[i,j]: the log delta t, or log shutter speed, for image j at pixel i
                (will be the same value for each i within the same j).
        l       lamdba, the constant that determines the amount of
                smoothness.
        w[z]:   the weighting function value for pixel value z (where z is between 0 - 255).

    Returns:
        g[z]:   the log exposure corresponding to pixel value z (where z is between 0 - 255).
        lE[i]:  the log film irradiance at pixel location i.

    """

    n = 256
    A = np.zeros((Z.shape[0]*Z.shape[1] + n + 1, n + Z.shape[0]))
    b = np.zeros((A.shape[0], 1))

    k = 1
    for i in np.arange(Z.shape[0]):
        for j in np.arange(Z.shape[1]):
            wij = w[Z[i,j]]
            A[k, Z[i,j]] = wij
            A[k, n+i] = -wij
            b[k, 0] = wij*B[i,j]
            k += 1

    A[k, 129] = 1
    k += 1

    for i in np.arange(n-2):
        A[k, i] = l*w[i]
        A[k, i+1] = -l*2*w[i]
        A[k, i+2] = l*w[i]
        k += 1

    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    print(U.shape, s.shape, Vh.shape)
    m = (abs(s)/np.max(abs(s))) > 1e-8
    U, s, Vh = U[:, m], s[m], Vh[m, :]
    x = Vh.conj().T @ np.diag(1/s) @ U.T @ b

    g = x[:n]
    lE = x[n:]

    return g, lE


def hdr(file_names, g_red, g_green, g_blue, w, exposure_matrix, nr_exposures):
    """
    Given the imaging system's response function g (per channel), a weighting function
    for pixel intensity values, and an exposure matrix containing the log shutter
    speed for each image, reconstruct the HDR radiance map in accordance to section
    2.2 of Debevec and Malik 1997.

    Args:
        file_names:           exposure stack image filenames
        g_red:                response function g for the red channel.
        g_green:              response function g for the green channel.
        g_blue:               response function g for the blue channel.
        w[z]:                 the weighting function value for pixel value z
                              (where z is between 0 - 255).
        exposure_matrix[i,j]: the log delta t, or log shutter speed, for image j at pixel i
                              (will be the same value for each i within the same j).
        nr_exposures:         number of images / exposures

    Returns:
        hdr:                  the hdr radiance map.
    """

    # read all the images, put into a numpy array
    images = []
    for filename in file_names:
        images.append(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB))
    images = np.array(images)
    # print(images[0].type)
    exposures = exposure_matrix[0,:]
    processed_images = []
    sum_weights =  np.zeros(images[0].shape)
    num = np.zeros(images[0].shape)
    for j, image in enumerate(images):
        r_w =  w[image[:,:,0]]
        g_w = w[image[:,:,1]]
        b_w = w[image[:,:,2]]
        
        r = g_red[image[:,:,0]]
        g = g_green[image[:,:,1]]
        b = g_blue[image[:,:,2]]
        weight = np.dstack([r_w, g_w, b_w])
        sum_weights += weight
        num += weight * (np.dstack([r, g, b]) - exposures[j])
        
    lnhdr = np.exp(num / sum_weights)
    lnhdr_minmax = (lnhdr - lnhdr.min()) / (lnhdr.max() - lnhdr.min())
    return lnhdr

# ========================================================================
# TONE MAPPING
# ========================================================================


def tm_global_simple(hdr_radiance_map):
    """
    Simple global tone mapping function (Reinhard et al.)

    Equation:
        E_display = E_world / (1 + E_world)

    Args:
        hdr_radiance_map (np.array): HDR radiance map of the image
                                     with shape (H, W, 3)
    Returns:
        np.array of image with values in range [0.0, 1.0]
    """
   
    unstretched = hdr_radiance_map / (1 + hdr_radiance_map)# globale scaling
    stretched = (unstretched - unstretched.min()) / (unstretched.max() - unstretched.min())
    return stretched



def tm_durand(hdr_radiance_map):
    """
    Your implementation of:
    http://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/DurandBilateral.pdf

    Args:
        hdr_radiance_map (np.array): HDR radiance map of the image
                                     with shape (H, W, 3)
    Returns:
        np.array of image with values in range [0.0, 1.0]
    """
    smallnumber = np.finfo(np.float32).tiny
    intensity = (hdr_radiance_map[:,:,0] * 0.2989 +  hdr_radiance_map[:,:,1] * 0.5870 +  hdr_radiance_map[:,:,2]*0.1140).astype(np.float32) # scaling for human perception of colors
    chrominance = hdr_radiance_map.astype(np.float32) / np.dstack([intensity, intensity,intensity])
    L = np.log2(intensity + smallnumber)
    B = cv2.bilateralFilter(L,  d=9, sigmaColor=75, sigmaSpace=75)
    D = L - B # detailed layer
    # plot the bilateral filter stuff
   
    dR = 5
    s = dR / (B.max() - B.min())
    B_prime = (B - B.max()) * s
    O = 2**(B_prime + D)
    result = (np.dstack([O , O ,O ]) * chrominance)**0.5
    # minmax
    (result - result.min()) / result.max()-result.min()
    return result

def bilateralshow(hdr_radiance_map):
    smallnumber = np.finfo(np.float32).tiny
    intensity = (hdr_radiance_map[:,:,0] * 0.2989 +  hdr_radiance_map[:,:,1] * 0.5870 + hdr_radiance_map[:,:,2]*0.1140).astype(np.float32) # scaling for human perception of colors
    chrominance = hdr_radiance_map.astype(np.float32) / np.dstack([intensity, intensity,intensity])
    L = np.log2(intensity + smallnumber)
    B = cv2.bilateralFilter(L,  d=9, sigmaColor=75, sigmaSpace=75)
    D = L - B # detailed layer
    # plot the bilateral filter stuff
    return L, B, D
   
