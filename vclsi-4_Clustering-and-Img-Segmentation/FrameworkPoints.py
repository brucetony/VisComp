import numpy as np
import scipy
from scipy import misc
import matplotlib
import  matplotlib.pyplot as plt
from matplotlib import cm


def get_point_edges(p, sigma_distance, edge_radius):
    """
    This function constructs a graph describing similarity of points in the given
       array.

    :param p: An array of shape (nPoint,2) where each row provides the
             coordinates of one of nPoint points in the plane.
    :param sigma_distance: The standard deviation of the Gaussian distribution used
             to weigh down longer edges.
    :param edge_radius: A positive float providing the maximal length of edges.
    :return:  tuple (edge_weight,edge_indices) where edge_weight is an array of
              length n_edge providing the weight of all produced edges and
              EdgeIndices is an integer array of shape (n_edge,2) where each row
              provides the indices of two pixels which are connected by an edge.
    """
    # Initialize lists
    weights = list()
    indices = list()

    # Iterate over points
    for i in range(len(p)):
        for j in range(i + 1, len(p)):

            # If less than edge radius then store indices an weights
            fx = ((p[i][0] - p[j][0]) ** 2) + ((p[i][1] - p[j][1]) ** 2)
            if fx ** 0.5 < edge_radius:
                c = np.exp(-(fx / (2 * (sigma_distance ** 2))))
                weights.append(c)
                indices.append((i, j))

    return weights, indices


def get_laplacian(n, weights, indices):
    """
    Constructs a matrix providing the Laplacian for the given graph.

    :param n: The number of vertices in the graph (resp. pixels in the image).
    :param weights: A one-dimensional array of nEdge floats providing the weight
             for each edge.
    :param indices: An integer array of shape (nEdge,2) where each row provides
             the vertex indices for one edge.
    :return: A matrix providing the Laplacian for the given graph.
    """
    # Empty matrix filled with zeros
    adjacency = np.zeros((n, n))
    degree = np.zeros((n, n))

    # Iterate over weights
    for k in range(len(weights)):
        adjacency[indices[k][0], indices[k][1]] = adjacency[indices[k][1], indices[k][0]] = weights[k]
        degree[indices[k][0], indices[k][0]] += weights[k]
        degree[indices[k][1], indices[k][1]] += weights[k]

    return degree - adjacency


def get_fiedler_vector(laplacian):
    """
    Given the Laplacian matrix of a graph this function computes the normalized
    Eigenvector for its second-smallest Eigenvalue (the so-called Fiedler vector)
    and returns it.

    :param laplacian: Laplacian matrix
    :return: laplacian matrix normalized by the Fiedler vector
    """

    return np.linalg.eigh(laplacian)[1][:, 1]

if (__name__ == "__main__"):
    # This list of points is to be clustered
    points = np.asarray(
        [(-8.097, 10.680), (-3.902, 8.421), (-9.711, 7.372), (0.859, 12.859), (4.732, 11.084), (-0.594, 9.147),
         (-4.224, 13.585), (-9.066, 11.891), (-13.181, 8.663), (-12.374, 3.983), (-11.406, -2.068), (-9.630, 2.854),
         (-13.665, -6.667), (-15.521, -0.454), (-15.117, -6.587), (-11.970, -10.621), (-6.000, -12.799),
         (-2.853, -14.978), (-8.501, -10.217), (2.311, -11.670), (3.441, -14.171), (5.861, -10.137), (10.138, -6.909),
         (15.382, -5.215), (14.091, 0.675), (11.187, 3.903), (8.685, 8.502), (7.879, 11.649), (5.216, 10.680),
         (11.025, 6.888), (13.446, 2.612), (12.962, -7.393), (8.363, -9.330), (-0.594, -0.212), (1.666, 1.401),
         (1.424, -1.019), (-0.351, -2.552), (-2.127, 0.675), (-0.271, 2.128), (-4.743, -4.016)])

    n_vertex = points.shape[0]

    # Construct the graph for the points
    edge_weight, edge_indices = get_point_edges(points, 1.0, 7.0)

    # Construct the Laplacian matrix for the graph
    laplacian = get_laplacian(n_vertex, edge_weight, edge_indices)

    # Compute the Fiedler vector
    fiedler_vector = get_fiedler_vector(laplacian)

    # Show the results
    plt.plot(list(range(0, len(fiedler_vector))), sorted(fiedler_vector), c="b")
    plt.show()

    for i, point in enumerate(points):
        if fiedler_vector[i] > -0.1:
            plt.scatter(point[0], point[1], color="b")
        else:
            plt.scatter(point[0], point[1], color="r")
    plt.show()