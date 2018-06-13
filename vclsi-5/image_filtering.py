import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import feature
from math import sqrt, exp, degrees, atan

plt.rcParams['image.cmap'] = 'gray'
# np.set_printoptions(threshold=np.nan)  # For showing whole numpy array

# Read image
brain_img = plt.imread('axial-brain.jpg')

# Create kernel 1
diag = [1/15] * 15
kernel_1 = np.diag(diag)  # Create diagonal matrix from above array

# Convolution filter image with kernel 1
convo_img_1 = ndimage.convolve(brain_img, kernel_1)
plt.imshow(convo_img_1)  # For jupyter notebook
# plt.show()

# Create kernel 2 and apply it
kernel_2 = np.zeros((15, 15))
kernel_2[7] = diag
convo_img_2 = ndimage.convolve(brain_img, kernel_2)
plt.imshow(convo_img_1)  # For jupyter notebook
# plt.show()

# Apply filters to img in different orders
convo_img_1then2 = ndimage.convolve(ndimage.convolve(brain_img, kernel_1), kernel_2)
convo_img_2then1 = ndimage.convolve(ndimage.convolve(brain_img, kernel_2), kernel_1)
plt.imshow(convo_img_1then2)
# plt.show()
plt.imshow(convo_img_2then1)
# plt.show()

# Part b
# Use Canny edge detection
canny_img1 = feature.canny(brain_img, sigma=0.001)
canny_img2 = feature.canny(brain_img, sigma=1)
canny_img3 = feature.canny(brain_img, sigma=2)
canny_img4 = feature.canny(brain_img, sigma=3)

# Display results
fig, ax = plt.subplots(2,2)

ax[0,0].imshow(canny_img1)
ax[0,0].axis('off')
ax[0,0].set_title('$\sigma=0.001$', fontsize=14)

ax[0,1].imshow(canny_img2)
ax[0,1].axis('off')
ax[0,1].set_title('$\sigma=1$', fontsize=14)

ax[1,0].imshow(canny_img3)
ax[1,0].axis('off')
ax[1,0].set_title('$\sigma=2$', fontsize=14)

ax[1,1].imshow(canny_img4)
ax[1,1].axis('off')
ax[1,1].set_title('$\sigma=3$', fontsize=14)

# fig.show()

# Apply filters to canny images and combine
# TODO Figure out what kind of filter

# Part c - Fish eye distortion

def fish_eye_distort(x, y, center):
    cx = center[0]
    cy = center[1]

    r0 = sqrt((x-cx)**2 + (y-cy)**2)
    ri = 0.87*exp((r0**(1/2.5))/1.5)
    thetai = atan((y-cy)/(x-cx))
    return ri, thetai

# TODO Convert polar to array coordinates. Degrees?
