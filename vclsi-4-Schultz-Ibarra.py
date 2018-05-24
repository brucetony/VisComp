import numpy as np
from skimage import color
from sklearn import mixture
from scipy import misc, ndimage
import matplotlib.pyplot as plt

# Read image into array data
raw_img_read = misc.imread("brain-noisy.png", True)

# Denoising image
mf_img = ndimage.median_filter(raw_img_read, size=5)  # What does this scalar size do exactly?
# misc.toimage(mf_img).show()  # Displays image

# Creating binary mask
binary_mask = mf_img > 0  # Any value greater than 0 (background)
bin_masked_img = mf_img.copy()
bin_masked_img[binary_mask] = 255  # 255 == white
# misc.toimage(bin_masked_img).show()  # Displays image

# Plot values from non-background pixels on a log scaled histogram
bins = 50
plt.gca().set_xscale("log")
counts, pixels, bars = plt.hist(mf_img[binary_mask], np.logspace(np.log10(10), np.log10(300), bins))
plt.xlabel("Pixel Value (log)")
plt.ylabel("Frequency")
plt.title("Brain image by pixel value")
# plt.show()  # Peaks refer to segmentation thresholds, gray/white matter and background

# Determine histogram peaks and the corresponding pixel value
peak_values = []
threshold = 75
for i in range(len(counts)-1):
    if counts[i] > threshold and counts[i] > counts[i-1] and counts[i] > counts[i+1]:
        peak_values.append(pixels[i])

#  Visualize image with peak pixel value locations after converting to RGB array
masks = []
pix_range = 40  # To give an acceptable range for pixel values
for pix_value in peak_values:
    masks.append(np.logical_and(pix_value+pix_range >= mf_img, mf_img >= pix_value-pix_range))

peak_img = bin_masked_img.copy()
peak_img = color.gray2rgb(peak_img)  # Convert to RGB array
prim_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # Define primary colors [R, G, B]
for counter, mask in enumerate(masks):
    if counter > 2:
        break
    peak_img[mask] = prim_colors[counter]

# misc.toimage(peak_img).show()  # Displays color image using peak values for masks

# Segmentation with Gaussian Mixture Model
# TODO create initializing points for the GMM - must have shape (3, 2)

# gmm_data = np.vstack([pixels[:-1], counts]).transpose()  # Combine histogram data in 2D array
gmm_data = np.column_stack(enumerate(mf_img[binary_mask])).transpose()  # Enumerate pixels with their grayscale values
gmm = mixture.GaussianMixture(n_components=3, means_init=peak_values)
gmm.fit(gmm_data)  # Estimate model parameters with the EM algorithm
group_colors = ['r', 'g', 'b']

# "Responsibility = conditional probability of point i belonging to cluster k
print(gmm.means_)



