import numpy as np
from skimage import color
from sklearn import mixture
from scipy import misc, ndimage
import matplotlib.pyplot as plt
from gaussian_mixture_model import *

np.set_printoptions(threshold=np.nan)  # For showing whole numpy array

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
plt.close()

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

########################################################################################################################
# sklearn GMM used here for learning
########################################################################################################################

# # Segmentation with Gaussian Mixture Model
# # Generate numpy array to initialize GMM with, use random pixel numbers and theorized peak values
# points_init = np.array([[1, 2, 3], peak_values]).transpose()
#
# gmm_data = np.column_stack(enumerate(mf_img[binary_mask])).transpose()  # Enumerate pixels with their grayscale values
# gmm = mixture.GaussianMixture(n_components=3, means_init=points_init)  # 3 clusters
# gmm.fit(gmm_data)  # Estimate model parameters with the EM algorithm
#
# # "Responsibility" = conditional probability of point i belonging to cluster k
# responsibilities = gmm.predict_proba(gmm_data)
# cluster_predict = gmm.predict(gmm_data)
#
# # Map responsibilities/cluster predictions to image
# prime_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # Define primary colors [R, G, B]
# gmm_img = bin_masked_img.copy()
# gmm_img = color.gray2rgb(gmm_img)

########################################################################################################################

# Create 2D array with pixel number (x value) and pixel intensity (y value)
gmm_data = np.column_stack(enumerate(mf_img[binary_mask])).transpose()
points_init = np.array([[1, 2, 3], peak_values]).transpose()

########################################################################################################################
# Homemade GMM with no optimization
########################################################################################################################

# Use homemade GMM functions to predict pixel clustering using only 1 iteration and no optimization
mix_coeff, sigma, means, responsibilities = GMM_convergence(gmm_data, 3,
                                                            iterations=iter, means_init=points_init, only_init=True)
cluster_predictions = [np.argmax(sample) for sample in responsibilities]
cluster_probabilities = [np.amax(sample) for sample in responsibilities]


def pixel_cluster_matcher(mask_template, cluster_assignment_list, cluster_number):
    """
    Uses a mask template to determine pixel location and iterates over new mask, changing Boolean\
    values to false if they don't match cluster_number
    :param mask_template: Mask_template to use to determine pixels of interest to change bool values
    :param cluster_assignment_list: 1D array with cluster assignment for every pixel that is True in mask_template
    :param cluster_number: Which cluster you are building this mask for
    :return: Mask with True values for only pixels at specified cluster_number location
    """
    new_mask = mask_template.copy()
    k = 0
    for pixel in np.nditer(new_mask, op_flags=['readwrite']):
        if pixel[...]:
            if cluster_assignment_list[k] != cluster_number:
                pixel[...] = False
            k += 1
    return new_mask


# Copy the binary mask image and convert to RGB
gmm_img = bin_masked_img.copy()
gmm_img = color.gray2rgb(gmm_img)

# Create masks for CSF, gray/white matter then assign them color layers
csf_mask = pixel_cluster_matcher(binary_mask, cluster_predictions, 0)
gray_mask = pixel_cluster_matcher(binary_mask, cluster_predictions, 1)
white_mask = pixel_cluster_matcher(binary_mask, cluster_predictions, 2)

gmm_img[csf_mask] = [255, 0, 0]
gmm_img[gray_mask] = [0, 255, 0]
gmm_img[white_mask] = [0, 0, 255]

# Multiply each value by the probability of that pixel belonging to that class (darker == less probable)
gmm_img[binary_mask] = [[value*cluster_probabilities[i] for value in pixel] for i, pixel in enumerate(gmm_img[binary_mask])]

# Display resulting image
# misc.toimage(gmm_img).show()  # Displays color image using peak values for masks

########################################################################################################################
# Homemade GMM with optimization until convergence
########################################################################################################################

# Using my homemade GMM algorithm until convergence
iter = 30
mix_coeff, sigma, means, responsibilities = GMM_convergence(gmm_data, 3, iterations=iter, means_init=points_init)

cluster_predictions = [np.argmax(sample) for sample in responsibilities]
cluster_probabilities = [np.amax(sample) for sample in responsibilities]

# Create masks for CSF, gray/white matter then assign them color layers
csf_mask = pixel_cluster_matcher(binary_mask, cluster_predictions, 0)
gray_mask = pixel_cluster_matcher(binary_mask, cluster_predictions, 1)
white_mask = pixel_cluster_matcher(binary_mask, cluster_predictions, 2)

gmm_img[csf_mask] = [255, 0, 0]
gmm_img[gray_mask] = [0, 255, 0]
gmm_img[white_mask] = [0, 0, 255]

# Multiply each value by the probability of that pixel belonging to that class (darker == less probable)
gmm_img[binary_mask] = [[value*cluster_probabilities[i] for value in pixel] for i, pixel in enumerate(gmm_img[binary_mask])]

# Display resulting image
misc.toimage(gmm_img).show()  # Displays color image using peak values for masks

# Create helper function to arrange values for plotting
def track_changes(value_array):
    # Standardize each iteration element
    differences = []
    for i, iteration in enumerate(value_array):
        value_array[i] = [value/sum(iteration) for value in iteration]
        if i > 0:
            differences.append([value_array[i][k]-value_array[i-1][k] for k in range(len(iteration))])
    return list(map(list, zip(*differences)))

# Graph the changes in model parameters with each iteration
# for i in range(len(track_changes(mix_coeff))):
#     plt.plot(range(iter), track_changes(mix_coeff)[i], color='red', label='Mixing Coefficients')
#     plt.plot(range(iter), track_changes(sigma)[i], color='blue', label='Variance')
#     plt.plot(range(iter), track_changes(means)[i], color='darkgreen', label='Means')
#     if i == 0:
#         plt.legend(['Mixing Coefficients', 'Variance', 'Means'])
# plt.axhline(y=0, color='black', linestyle='-')
# plt.xlabel('EM Iteration')
# plt.ylabel('Cluster parameter difference from previous iteration')
# plt.title('Model Parameter Changes per Iteration (for each cluster)')
# plt.show()

# TODO also map probabilities? Use as a scalar?


# misc.imsave('GMM_image.png', gmm_img)  # Save output image for GMM computation


