from random import randint
from math import pi, exp, sqrt
import numpy as np

def GMM_init(data_points, n_distributions, means_init=None):

    mix_coeff = [1/n_distributions] * n_distributions  # Sum of pi across all clusters must = 1
    if means_init is not None:
        means = means_init[:,1]  # Set initial means to user specified values
    else:
        # Random initialization using y values ([:,1])
        means = [randint(min(data_points[:,1]), max(data_points[:,1])) for i in range(n_distributions)]

    # Initialize variance to sig**2 = sum(X-mu)**2 / N
    init_variance = sum([(data_points[i,1]-min(means))**2 for i in range(len(data_points[:,1]))])
    sigma = [sqrt(init_variance/len(data_points[:,1]))] * n_distributions
    return mix_coeff, sigma, means

# TODO implement k-means init
# TODO implement membership vector z

# E-step of GMM algorithm
def GMM_responsibilities(data_points, n_distributions, mix_coeff, sigma, means):

    # Calculate gaussians
    # GMM array has x values in first column and GMM sum value in the last column
    GMM_array = np.empty((len(data_points[:, 1]), n_distributions + 2))
    GMM_array[:, 0] = data_points[:, 1]  # First column is our values
    for i in range(len(data_points[:, 1])):  # Iterate through values
        for k in range(n_distributions):  # Iterate through clusters
            gauss = 1/(sqrt(2*pi)*sigma[k])*exp(-((data_points[i, 1]-means[k])**2)/(2*(sigma[k]**2)))
            GMM_array[i, k+1] = mix_coeff[k]*gauss
        GMM_array[i][n_distributions + 1] = sum(GMM_array[i][1:n_distributions+1])  # Sum N(x|uk, sigk**2) (Gauss_dis) values

    # Calculate responsibilities
    responsibilities = np.empty((len(data_points[:, 1]), n_distributions))
    for i in range(len(data_points[:, 1])):
        for k in range(n_distributions):
            responsibilities[i][k] = GMM_array[i, k+1]/GMM_array[i, n_distributions+1]

    # Only responsibilities values! i (sample #) rows by k (cluster #) columns
    return responsibilities


# M-step of GMM algorithm
def GMM_optimize(data_points, n_distributions, mix_coeff, sigma, means):

    rho = GMM_responsibilities(data_points, n_distributions, mix_coeff, sigma, means)

    # Create lists to fill with optimized values
    opt_mix_coeff = [0] * len(mix_coeff)
    opt_means = [0] * len(means)
    opt_sigma = [0] * len(sigma)

    # Optimize parameters
    for k in range(n_distributions):
        cluster_resp_sum = sum(rho[:, k])

    # Mixing Coefficients
        opt_mix_coeff[k] = cluster_resp_sum / len(data_points[:, 1])

    # Means
        mean_numerator = sum([rho[i][k]*data_points[i][1] for i in range(len(data_points[:,1]))])
        opt_means[k] = mean_numerator/cluster_resp_sum

    # Sigma
        sig_numerator = sum([(rho[i][k]*((data_points[i][1]-means[k])**2)) for i in range(len(data_points[:, 1]))])
        opt_sigma[k] = sqrt(sig_numerator/cluster_resp_sum)

    return opt_mix_coeff, opt_sigma, opt_means, rho


def GMM_convergence(data_points, n_distributions, iterations=50, means_init=None):
    mix_coeff, sigma, means = GMM_init(data_points, n_distributions, means_init)

    # Create list to track changes with every iteration
    mix_coefficient_list = [list(mix_coeff)]
    sigma_list = [list(sigma)]
    means_list = [list(means)]

    i = 0
    while i < iterations:
        mix_coeff, sigma, means, rho = GMM_optimize(data_points, n_distributions, mix_coeff, sigma, means)
        mix_coefficient_list.append(mix_coeff)
        sigma_list.append(sigma)
        means_list.append(means)
        i += 1
    return mix_coefficient_list, sigma_list, means_list, rho

# TODO Make this iterate and fix functions to work together better -- use OOP?

# foo = np.array([[1, 2], [1, 3], [3, 6], [7, 7]])
# bar, dad, balls = GMM_convergence(foo, 2, iterations=5)
# print(balls)

