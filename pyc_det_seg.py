
#import required packages and functions

import matplotlib.pyplot as plt
import numpy as np
from skimage import util
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage import io, color
from skimage import morphology
import cv2
from skimage.morphology import extrema
from skimage.measure import label
from skimage import exposure
import os
import scipy.ndimage as ndimage
import imageio

#function to convert to single channel gray-scale image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

path_folder_raw = "O:/FIP/2018/WW023/RefTraits/Macro/stb_senescence2018_fpww023/macro_outcomes"

#function to list all images to analyze
def list_files(dir):
    file_list = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        for f in filenames:
            if f.endswith("leafOriginal.png"):
                file_list.append(os.path.join(dirpath, f))
    return file_list

#list all images to analyze
files = list_files(path_folder_raw)

#iterate of all images
for i in files:

    print(i)



########################################################################################################################
# Segment leaf from background
########################################################################################################################

# Load image, convert from BGR to RGB
img = cv2.cvtColor(cv2.imread(r"O:\FIP\2018\WW023\RefTraits\Macro\stb_senescence2018_fpww023\macro_outcomes\t4\Overlay\fpww023_t4_sn4_8_leafOriginal.png"),
                   cv2.COLOR_BGR2RGB)

#"Feature Maps"

feature_map = img[1050:1450, 4600:5100]

#plot
fig, ax = plt.subplots()
ax.plot()
ax.imshow(feature_map)
ax.set_title('Quickshift segmentation')
for a in ax.ravel():
    a.set_axis_off()
plt.tight_layout()
plt.show()

imageio.imwrite(r"O:\Projects\KP0011\3\RefData\Test_py\fmaps\fpww023_t4_sn4_8_leafOriginal5.png", feature_map)



#crop to area of interest, removing black lines
img = img[350:1900, 235:8200]

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
min_size = 1000000

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

imageio.imwrite(r"O:\Projects\KP0011\3\RefData\Test_py\fpww023_t4_sn955_2_leafOriginal.png", Color_Masked)

########################################################################################################################
# Transform to grayscale image
########################################################################################################################

#convert
img_gs = rgb2gray(Color_Masked)

#invert
img_gs_inv = util.invert(img_gs)
img_gs_inv = np.around(img_gs_inv, decimals=0)

########################################################################################################################
# Detect Pycnidia
########################################################################################################################

#prepare image for h_maxima and local_maxima functions
img = color.rgb2gray(Color_Masked)
img = 255-img
img = exposure.rescale_intensity(img)

#extract all local maxima
local_maxima = extrema.local_maxima(img)
label_maxima = label(local_maxima)
overlay = color.label2rgb(label_maxima, img, alpha=0.7, bg_label=0,
                          bg_color=None, colors=[(1, 0, 0)])

#extract local maxima with certrain regional contrast
h = 0.05
nb = morphology.selem.disk(radius=3)
h_maxima = extrema.h_maxima(img, h, selem=nb)
label_h_maxima = label(h_maxima)
overlay_hr = color.label2rgb(label_h_maxima, Color_Masked, alpha=0.7, bg_label=0,
                            bg_color=None, colors=[(1, 0, 0)])

#plot results
fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax[0].imshow(Color_Masked, cmap='gray', interpolation='none')
ax[0].set_title('Original image')
ax[0].axis('off')
ax[1].imshow(overlay, interpolation='none')
ax[1].set_title('Local Maxima')
ax[1].axis('off')
ax[2].imshow(overlay_hr, interpolation='none')
ax[2].set_title('h maxima for h = %.2f' % h)
ax[2].axis('off')
plt.show()

########################################################################################################################
# Segmentation into Super-Pixels
########################################################################################################################

#get leaf size in pixels
leaf_size = np.sum(img_gs != 255)

#adjust segments to leaf size
#sqrt seems to work well, to avoid unduly high number for large leaves
n_segs = 3*np.sqrt(leaf_size)

#segment in Super-Pixels
segments_slic_ok = slic(Color_Masked, n_segments=n_segs, compactness=1, sigma=15, max_size_factor=3)

#plot
fig, ax = plt.subplots()
ax.plot()
ax.imshow(mark_boundaries(Color_Masked, segments_slic_ok))
ax.set_title('Quickshift segmentation')
for a in ax.ravel():
    a.set_axis_off()
plt.tight_layout()
plt.show()

########################################################################################################################
