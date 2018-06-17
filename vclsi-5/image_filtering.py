import numpy as np
from scipy import ndimage
from skimage import feature
import matplotlib.pyplot as plt
from math import sqrt, exp, atan

plt.rcParams['image.cmap'] = 'gray'
np.set_printoptions(threshold=np.nan)  # For showing whole numpy array

# Read image
brain_img = plt.imread('axial-brain.jpg')

# Create kernel 1
diag = [1/15] * 15
kernel_1 = np.diag(diag)  # Create diagonal matrix from above array

# Convolution filter image with diagonal kernel - kernel 1
convo_img_1 = ndimage.convolve(brain_img, kernel_1)
plt.imshow(convo_img_1)  # For jupyter notebook
# plt.show()

# Create kernel 2 and apply it
kernel_2 = np.zeros((15, 15))
kernel_2[7] = diag
convo_img_2 = ndimage.convolve(brain_img, kernel_2)
plt.imshow(convo_img_2)  # For jupyter notebook
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
plt.close()

# Apply filters to canny images and combine
canny_composite = np.zeros(brain_img.shape)  # Copy original image (just for shape)
canny_composite[canny_img1] = 25
canny_composite[canny_img2] = 75
canny_composite[canny_img3] = 125
canny_composite[canny_img4] = 255
plt.imshow(canny_composite)
plt.axis('off')
plt.title('Canny composite image')
# plt.show()

# Part c - Fish eye distortion


def fish_eye(img_input):
    width, height = img_input.shape
    distorted_img = np.zeros(img_input.shape)
    cx, cy = width/2, height/2
    for y_coord in range(len(img_input)):
        for x_coord in range(len(y_coord)):
            distorted_img[y_coord][x_coord] = 2
    r_out = sqrt((x-cx)**2 + (y-cy)**2)
    r_in = 0.87*exp((r_out**(1/2.5))/1.5)
    theta_in = atan((y-cy)/(x-cx))
    return r_in, theta_in


# TODO Convert polar to array coordinates.
