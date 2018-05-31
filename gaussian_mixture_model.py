from random import randint
from math import pi, exp, sqrt
import numpy as np

def GMM_init(data_points, n_distributions, means_init=None):
    sigma = [1] * n_distributions  # Initialize sigma to 1 for each cluster
    mix_coeff = [1/n_distributions] * n_distributions  # Sum of pi across all clusters must = 1
    if means_init:
        means = means_init  # Set initial means to user specified values
    else:
        # Random initialization using y values ([:,1])
        means = [randint(min(data_points[:,0]), max(data_points[:,0])) for i in range(n_distributions)]

    GMM = np.empty((len(data_points[:,0]), n_distributions+2))  # One for x values, and one for each cluster value and one sum values
    GMM[:,0] = data_points[:,0]  # First column is our x values
    for i in range(len(data_points[:, 0])):
        for k in range(n_distributions):
            GMM[i, k+1] = mix_coeff[k]*1/((sqrt(2*pi)*sigma[k])*exp(-(data_points[i,1]-means[k])**2/(2*(sigma[k]**2))))
        GMM[i][n_distributions+1] = sum(GMM[i][1:])  # Sum N(x|uk, sigk**2) (Gauss_dis) values

    # GMM array has x values in first column and GMM sum value in the last column
    return GMM, mix_coeff, sigma, means

# TODO implement membership vector z

# E-step of GMM algorithm
def GMM_responsibilities(data_points, n_distributions, means_init=None):
    GMM, mix_coeff, sigma, means = GMM_init(data_points, n_distributions, means_init)
    responsibilities = np.empty((len(data_points[:,0]), n_distributions))
    for i in range(len(data_points[:, 0])):
        for k in range(n_distributions):
            responsibilities[i][k] = GMM[i, k+1]/GMM[i, n_distributions+1]

    # Only resp values! i (sample #) rows by k (cluter #) columns
    return responsibilities

# M-step of GMM algorithm
def GMM_optimize(data_points, n_distributions, means_init=None):
    GMM, mix_coeff, sigma, means = GMM_init(data_points, n_distributions, means_init)
    rho = GMM_responsibilities(data_points, n_distributions, means_init)

    # Means
    for k in range(n_distributions):
        mean_numerator = sum([rho[i][k]*data_points[i][0] for i in range(len(data_points[:,0]))])
        cluster_resp_sum = sum([val for val in rho[:,k]])
        means[k] = mean_numerator/cluster_resp_sum

    # Sigma
        sig_numerator = sum([rho[i][k]*(data_points[i][0]-means[k])**2 for i in range(len(data_points[:, 0]))])
        sigma[k] = sqrt(sig_numerator/cluster_resp_sum)

    # Mixing Coefficients
        mix_coeff[k] = cluster_resp_sum/len(data_points[:, 0])

    return means, sigma, mix_coeff

# TODO Make this iterate and fix functions to work together better -- uses OOP?

foo = np.array([[5, 2], [1, 9], [3, 2], [7, 12]])
bar, dad, balls = GMM_optimize(foo, 2)
print(dad)