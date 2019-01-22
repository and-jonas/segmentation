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
img = cv2.cvtColor(cv2.imread(r"O:\Projects\KP0011\3\RefData\Result\Test\fpww023_t4_sn2_1_leafOriginal.png"),
                   cv2.COLOR_BGR2RGB)

#remove white background
lower_white = np.array([180, 200, 200], dtype=np.uint8) #threshold for white pixels
upper_white = np.array([255, 255, 255], dtype=np.uint8)
mask1 = cv2.inRange(img, lower_white, upper_white)  # could also use threshold
mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # "erase" the small white points in the resulting mask
mask1 = cv2.bitwise_not(mask1)  # invert mask
#remove black background
lower_black = np.array([0, 0, 0], dtype=np.uint8)
upper_black = np.array([120, 120, 120], dtype=np.uint8) #threshold for black pixels
mask2 = cv2.inRange(img, lower_black, upper_black)  # could also use threshold
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # "erase" the small white points in the resulting mask
mask2 = cv2.bitwise_not(mask2)  # invert mask
#combine masks
mask = mask1 + mask2
mask = morphology.remove_small_objects(mask, min_size = 75)
mask = cv2.blur(mask, ksize = (35, 35))

[indx, indy] = np.where(mask == 255)

Color_Masked = img.copy()
Color_Masked[indx, indy] = 255

########################################################################################################################
# Transform to grayscale image
########################################################################################################################

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img_gs = rgb2gray(Color_Masked)

#invert grayscale image
img_gs_inv = util.invert(img_gs)
img_gs_inv = np.around(img_gs_inv, decimals=0)

########################################################################################################################
# Detect Pycnidia
########################################################################################################################

#detect all local maxima
local_maxima = extrema.local_maxima(img_gs_inv)
label_maxima = label(local_maxima)
overlay = color.label2rgb(label_maxima, img_gs_inv, alpha=0.7, bg_label=0,
                          bg_color=None, colors=[(1, 0, 0)])

#detect local maxima with a height of h
#which is a measure of local contrast (gray value level to descend)
h = 0.05
#we are looking for roughly circular maxima
form = morphology.selem.disk(radius=3)
h_maxima = extrema.h_maxima(img, h, selem=form)
label_h_maxima = label(h_maxima)
overlay_h = color.label2rgb(label_h_maxima, img, alpha=0.7, bg_label=0,
                            bg_color=None, colors=[(1, 0, 0)])

#plot
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
