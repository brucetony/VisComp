import numpy as np
from skimage import color
from sklearn import mixture
from scipy import misc, ndimage
import matplotlib.pyplot as plt

# np.set_printoptions(threshold=np.nan)  # For showing whole numpy array

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
prime_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # Define primary colors [R, G, B]
for counter, mask in enumerate(masks):
    if counter > 2:
        break
    peak_img[mask] = prime_colors[counter]

# misc.toimage(peak_img).show()  # Displays color image using peak values for masks

# Segmentation with Gaussian Mixture Model
# Generate numpy array to initialize GMM with, use random pixel numbers and theorized peak values
points_init = np.array([[1, 2, 3], peak_values]).transpose()

gmm_data = np.column_stack(enumerate(mf_img[binary_mask])).transpose()  # Enumerate pixels with their grayscale values
gmm = mixture.GaussianMixture(n_components=3, means_init=points_init)  # 3 clusters
gmm.fit(gmm_data)  # Estimate model parameters with the EM algorithm

# "Responsibility" = conditional probability of point i belonging to cluster k
responsibilities = gmm.predict_proba(gmm_data)
cluster_predict = gmm.predict(gmm_data)


# Map responsibilities/cluster predictions to image
prime_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # Define primary colors [R, G, B]
gmm_img = bin_masked_img.copy()
gmm_img = color.gray2rgb(gmm_img)

csf_mask = cluster_predict == 0
gray_mask = cluster_predict == 1
white_mask = cluster_predict == 2

# TODO figure out how to revert a linear array back into an image array

# for counter in range(len(gmm_img[binary_mask])):
#     if cluster_predict[counter] == 0:
#         gmm_img[binary_mask][counter] = [255, 0, 0]  # Red
#     elif cluster_predict[counter] == 1:
#         gmm_img[binary_mask][counter] = [0, 255, 0]  # Green
#     elif cluster_predict[counter] == 2:
#         gmm_img[binary_mask][counter] = [0, 0, 255]  # Blue


# misc.toimage(gmm_img).show()  # Displays GMM colored image

# gmm.means_[:,1]
