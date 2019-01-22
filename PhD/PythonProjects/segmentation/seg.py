#import required packages and functions
import matplotlib.pyplot as plt
import numpy as np
import imageio
from skimage import util
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io, color
from skimage import morphology
import cv2
from functools import reduce
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from PIL import Image
import PIL.ImageOps
from scipy import ndimage as ndi
from skimage.morphology import extrema
from skimage.measure import label
from skimage import exposure
import os

path_folder_raw = "O:/FIP/2018/WW023/RefTraits/Macro/stb_senescence2018_fpww023/macro_outcomes"

#function to list all images to analyze
def list_files(dir):
    file_list = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        for f in filenames:
            if f.endswith("leafOriginal.png"):
                if "_t3_" in f:
                    file_list.append(os.path.join(dirpath, f))
    return file_list

#list all images to analyze
files = list_files(path_folder_raw)

save_path = "O:/FIP/2018/WW023/RefTraits/Preprocessed/t3/"

#iterate over images
for k in files:

    try:

        # Load image, convert from BGR to RGB
        img = cv2.cvtColor(cv2.imread(k),
                           cv2.COLOR_BGR2RGB)
        img.shape  # a three dimensional array

        # crop to area of interest, removing black lines
        img = img[350:1900, 275:8200]

        # remove white background
        # blur image a bit
        blur = cv2.GaussianBlur(img, (15, 15), 2)

        # mask for paper background
        lower_white = np.array([200, 200, 200], dtype=np.uint8)  # threshold for white pixels
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(blur, lower_white, upper_white)  # could also use threshold
        # mask needs to be inverted,
        # since we want to set the BACKGROUND to white
        mask1 = cv2.bitwise_not(mask1)

        # There are still spots not belonging to the leaf
        # remove small objects to get rid of them

        # find all connected components
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask1, connectivity=8)
        # connectedComponentswithStats yields every seperated component with information on each of them, such as size
        # take out the background which is also considered a component
        sizes = stats[1:, -1];
        nb_components = nb_components - 1

        # minimal size of particle
        # somwhere between largest unreal feature and leaf size
        min_size = 500000

        # cleaned mask
        mask_cleaned = np.zeros((output.shape))
        # for every component in the image,
        # keep only those above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                mask_cleaned[output == i + 1] = 255

        # apply cleaned mask to the image
        [indx, indy] = np.where(mask_cleaned == 0)
        Color_Masked = img.copy()
        Color_Masked[indx, indy] = 255

        # Transform to HSV
        img = cv2.cvtColor(Color_Masked, cv2.COLOR_RGB2HSV)

        # Create a mask for brown pixels
        mask = cv2.inRange(img, np.array([0, 95, 95]),np.array([30, 255, 255]))


        ####

        img_fill_holes = ndi.binary_fill_holes(cv2.bitwise_not(mask), structure=np.ones((4,4))).astype(np.uint8)
        img_fill_holes = cv2.floodFill(img_fill_holes, mask, (0, 0), 255);

        ####


        # Fill holes in the mask

        img_fill_holes = ndi.binary_fill_holes(mask, structure=np.ones((20,20))).astype(np.uint8)

        # Remove Noise
        ## Rectangular Kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(6,6))
        ## Remove noise by morphological opening
        opening = cv2.morphologyEx(img_fill_holes, cv2.MORPH_OPEN, kernel)

        # Remove small areas
        ## Find all connected components
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8)
        ## ConnectedComponentswithStats yields every seperated component with information on each of them, such as size
        ## Take out the background which is also considered a component
        sizes = stats[1:, -1]; nb_components = nb_components - 1
        ## Define minimal size of particle
        min_size = 10001
        ## Create cleaned mask
        mask_cleaned = np.zeros((output.shape))
        ## for every component in the image,
        ## keep only those above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                mask_cleaned[output == i + 1] = 255
        mask_cleaned = np.uint8(mask_cleaned)

        # Find contours
        _, contours, _ = cv2.findContours(mask_cleaned, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours onto original image
        cnt = cv2.drawContours(Color_Masked, contours, -1, (128,255,0), 2)

        # plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
        ax[0].imshow(mask)
        ax[0].set_title("dcs")
        ax[1].imshow(img_fill_holes)
        ax[1].set_title("original")
        plt.tight_layout()
        plt.show()

        cnt = cv2.resize(cnt, (0, 0), fx=0.5, fy=0.5)


        # Save overlay
        filename = os.path.basename(k)
        cv2.imwrite(save_path + filename, cv2.cvtColor(cnt, cv2.COLOR_RGB2BGR))

    except:

        print("Error in: " + k)

########################################################################################################################
# Segment leaf from background
########################################################################################################################

# Load image, convert from BGR to RGB
img = cv2.cvtColor(cv2.imread(r"O:\Projects\KP0011\3\RefData\Test_py\fpww023_t5_fungic_sn127_5_leafOriginal.png"),
                   cv2.COLOR_BGR2RGB)
img.shape #a three dimensional array

#crop to area of interest, removing black lines
img = img[350:1900, 270:8200]

#remove white background
#blur image a bit
blur = cv2.GaussianBlur(img, (15, 15), 2)

#mask for paper background
lower_white = np.array([200, 200, 200], dtype=np.uint8) #threshold for white pixels
upper_white = np.array([255, 255, 255], dtype=np.uint8)
mask1 = cv2.inRange(blur, lower_white, upper_white)  # could also use threshold
#mask needs to be inverted,
#since we want to set the BACKGROUND to white
mask1 = cv2.bitwise_not(mask1)

#There are still spots not belonging to the leaf
#remove small objects to get rid of them

#find all connected components
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask1, connectivity=8)
#connectedComponentswithStats yields every seperated component with information on each of them, such as size
#take out the background which is also considered a component
sizes = stats[1:, -1]; nb_components = nb_components - 1

#minimal size of particle
#somwhere between largest unreal feature and leaf size
min_size = 500000

#cleaned mask
mask_cleaned = np.zeros((output.shape))
#for every component in the image,
#keep only those above min_size
for i in range(0, nb_components):
    if sizes[i] >= min_size:
        mask_cleaned[output == i + 1] = 255

#apply cleaned mask to the image
[indx, indy] = np.where(mask_cleaned == 0)
Color_Masked = img.copy()
Color_Masked[indx, indy] = 255

#plot
fig, ax = plt.subplots(1,1, figsize=(10, 10))
ax.imshow(Color_Masked)
ax.set_title("MASKED")
plt.tight_layout()
plt.show()

########################################################################################################################
# De-correlation stretching
########################################################################################################################

#code source: https://github.com/lbrabec/decorrstretch
def decorrstretch(A, tol=None):
    """
    Apply decorrelation stretch to image
    Arguments:
    A   -- image in cv2/numpy.array format
    tol -- upper and lower limit of contrast stretching
    """

    # save the original shape
    orig_shape = A.shape
    # reshape the image
    #         B G R
    # pixel 1 .
    # pixel 2   .
    #  . . .      .
    A = A.reshape((-1,3)).astype(np.float)
    # covariance matrix of A
    cov = np.cov(A.T)
    # source and target sigma
    sigma = np.diag(np.sqrt(cov.diagonal()))
    # eigen decomposition of covariance matrix
    eigval, V = np.linalg.eig(cov)
    # stretch matrix
    S = np.diag(1/np.sqrt(eigval))
    # compute mean of each color
    mean = np.mean(A, axis=0)
    # substract the mean from image
    A -= mean
    # compute the transformation matrix
    T = reduce(np.dot, [sigma, V, S, V.T])
    # compute offset
    offset = mean - np.dot(mean, T)
    # transform the image
    A = np.dot(A, T)
    # add the mean and offset
    A += mean + offset
    # restore original shape
    B = A.reshape(orig_shape)
    # for each color...
    for b in range(3):
        # apply contrast stretching if requested
        if tol:
            # find lower and upper limit for contrast stretching
            low, high = np.percentile(B[:,:,b], 100*tol), np.percentile(B[:,:,b], 100-100*tol)
            B[B<low] = low
            B[B>high] = high
        # ...rescale the color values to 0..255
        B[:,:,b] = 255 * (B[:,:,b] - B[:,:,b].min())/(B[:,:,b].max() - B[:,:,b].min())
    # return it as uint8 (byte) image
    return B.astype(np.uint8)

mask_ds = decorrstretch(Color_Masked)

#plot
fig, ax = plt.subplots(1,2, figsize=(10, 10),  sharex=True, sharey=True)
ax[0].imshow(mask_ds)
ax[0].set_title("dcs")
ax[1].imshow(Color_Masked)
ax[1].set_title("original")
plt.tight_layout()
plt.show()

########################################################################################################################
# Transform to grayscale image
########################################################################################################################

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img_gs = rgb2gray(Color_Masked)

fig, ax = plt.subplots(1,1, figsize=(10, 10))
ax.imshow(img_gs, cmap = "gray")
ax.set_title("gray")
plt.tight_layout()
plt.show()

#invert grayscale image
img_gs_inv = util.invert(img_gs)
img_gs_inv = np.around(img_gs_inv, decimals=0)

fig, ax = plt.subplots()
ax.plot()
ax.imshow(img_gs_inv, cmap = "gray")
ax.set_title('mask')
plt.tight_layout()
plt.show()

########################################################################################################################
# Detect Pycnidia
########################################################################################################################

# image_max is the dilation of im with a 20*20 structuring element
# It is used within peak_local_max function
image_max = ndi.maximum_filter(img_gs_inv, size=3, mode='constant')

# Comparison between image_max and im to find the coordinates of local maxima
coordinates = peak_local_max(img_gs_inv, min_distance=3, threshold_abs=-250)
# display result
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(Color_Masked, cmap="gray")
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(img_gs_inv, cmap="gray")
ax[1].autoscale(False)
ax[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
ax[1].axis('off')
ax[1].set_title('Peak local max')

fig.tight_layout()

plt.show()

###########################################

#prepare image for h_maxima and local_maxima functions
img = color.rgb2gray(Color_Masked)
img = 255-img
img = exposure.rescale_intensity(img)

#extract local maxima
local_maxima = extrema.local_maxima(img_gs_inv)
label_maxima = label(local_maxima)
overlay = color.label2rgb(label_maxima, img_gs_inv, alpha=0.7, bg_label=0,
                          bg_color=None, colors=[(1, 0, 0)])

#local maxima with certrain contrast
h = 0.05
ding = morphology.selem.disk(radius=3)
h_maxima = extrema.h_maxima(img, h, selem=ding)
label_h_maxima = label(h_maxima)
overlay_h = color.label2rgb(label_h_maxima, img, alpha=0.7, bg_label=0,
                            bg_color=None, colors=[(1, 0, 0)])

# a new figure with 3 subplots
fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

ax[0].imshow(Color_Masked, cmap='gray', interpolation='none')
ax[0].set_title('Original image')
ax[0].axis('off')

ax[1].imshow(overlay, interpolation='none')
ax[1].set_title('Local Maxima')
ax[1].axis('off')

ax[2].imshow(overlay_h, interpolation='none')
ax[2].set_title('h maxima for h = %.2f' % h)
ax[2].axis('off')
plt.show()

########################################################################################################################
# Segmentation into Super-Pixels
########################################################################################################################

#get leaf size in pixels
leaf_size = np.sum(img_gs != 255)

#adjust segments to leaf size
nsegs = leaf_size/150

segments_slic_ok = slic(Color_Masked, n_segments=nsegs, compactness=10, sigma=1, max_size_factor=3)
fig, ax = plt.subplots()
ax.plot()
ax.imshow(mark_boundaries(Color_Masked, segments_slic_ok))
ax.set_title('qs')
for a in ax.ravel():
    a.set_axis_off()
plt.tight_layout()
plt.show()




###########################################


crop_img = Color_Masked[1250:1600, 200:3000]

fig, ax = plt.subplots()
ax.plot()
ax.imshow(img)
ax.set_title('mask')
plt.tight_layout()
plt.show()


segments_slic_ok = slic(crop_img, n_segments=1000, compactness=10, sigma=1, max_size_factor=3)
segments_slic = slic(crop_img, n_segments=1000, compactness=10, sigma=0.75, max_size_factor=3)


segments_quick100 = quickshift(crop_img, kernel_size=5, max_dist=100, ratio=1)
segments_quick200 = quickshift(crop_img, kernel_size=5, max_dist=200, ratio=1)
segments_quick400 = quickshift(crop_img, kernel_size=5, max_dist=400, ratio=1)
segments_quick800 = quickshift(crop_img, kernel_size=5, max_dist=800, ratio=1)
segments_quick2 = quickshift(crop_img, kernel_size=8, max_dist=800, ratio=1)

gradient = sobel(rgb2gray(crop_img))
segments_watershed = watershed(gradient, markers=250, compactness=0.001)

segments_fz = felzenszwalb(crop_img, scale=50, sigma=1, min_size=50)

fig, ax = plt.subplots()
ax.plot()
ax.imshow(mark_boundaries(crop_img, segments_fz))
ax.set_title('qs')
for a in ax.ravel():
    a.set_axis_off()
plt.tight_layout()
plt.show()

###########################################








