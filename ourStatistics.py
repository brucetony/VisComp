import numpy as np
import pandas as pd
from sklearn import manifold
from sklearn.decomposition import PCA

def interpolate_by_mean(column):
    """
    Interpolates missing values by the mean
    :param column: A pandas dataframe
    :return: A pandas dataframe containing the column names which have null values
    """
    return column.fillna(int(round(column.mean()))) if any(column.isnull()) else column


def get_f_score(attribute, groups, exclude=[]):
    """
    Calculates the F score of an attribute between to groups (for now)
    :param data: Pandas dataframe
    :param column: Pandas Series
    :param groupby: Column with the categories
    :param exclude: List of columns to be excluded from the calculation
    """
    if attribute.name in exclude:
        return None
    else:
        # Get means for groups
        grand_mean = attribute.mean()
        g1_mean = groups[0][attribute.name].mean()  # Group 1
        g2_mean = groups[1][attribute.name].mean()  # Group 2

        g1_diff = [(i - g1_mean) ** 2 for i in groups[0][attribute.name]]  # Group 1
        g2_diff = [(i - g2_mean) ** 2 for i in groups[1][attribute.name]]  # Group 2

        numerator = (g1_mean - grand_mean) ** 2 + (g2_mean - grand_mean) ** 2
        denom = (1 / (len(groups[0][attribute.name]) - 1)) * sum(g1_diff) + \
                (1 / (len(groups[1][attribute.name]) - 1)) * sum(g2_diff)

        return numerator / denom

def cluster_center(points):
    '''
    Computes the center point from a list of points
    :param points: Can be a list of lists of tuples containing points
    :return: A list with center point for each set of points provided, in numpy array format
    '''
    cluster_centers = [np.array([0, 0]) for dataset in points]
    for i in range(len(points)):
        for datapoint in points[i]:
            cluster_centers[i] = np.add(np.array(datapoint), cluster_centers[i])
        cluster_centers[i] = cluster_centers[i]/len(points[i])
    return cluster_centers

def DSC(point_list):
    '''
    Computers Distance consistency for a set (or sets) of points
    :param point_list: A list of points in tuple format
    :return: Numerical computation of the Distance consistency for all points provided
    '''
    # First computer center points
    centers = cluster_center(point_list)
    #Create empty lists to store distances of the difference between each point and each centroid
    same_cluster_distances = []
    diff_cluster_distances = []
    for i in range(len(point_list)):
        for datapoint in point_list[i]:
            for j in range(len(point_list)):
                # Alternate between same centroid and diff centroids, store in list
                if i == j:
                    same_cluster_distances.append(np.linalg.norm(np.array(datapoint) - centers[i]))
                else:
                    diff_cluster_distances.append(np.linalg.norm(np.array(datapoint) - centers[j]))
    # Create counter to keep track points closer to their own centroid
    closer_points = 0
    for k in range(len(same_cluster_distances)):
        if same_cluster_distances[k] < diff_cluster_distances[k]:
            closer_points += 1
    return (100*closer_points/len(same_cluster_distances))


def runPCA(data, exclude=[]):
    '''
    Perform PCA on a dataset
    :param data: Pandas dataset
    :param exclude: columns to exclude during PCA analysis
    :return:
    '''

    # Drop excluded columns (non-PCA features)
    data = data.drop(columns=exclude)

    # Initialize PCA class with default values
    pca = PCA()

    # Get scores of PCA (Fit PCA)
    scores = pca.fit_transform(data)

    # Get loadings
    loadings = pca.components_

    # Get aditional information out of PCA (summary)
    sd = scores.std(axis=0)
    var = pca.explained_variance_ratio_
    cumVar = var.cumsum()

    # Create summay file
    summary = np.array([sd, var, cumVar]).T

    # Create headers
    header = ["PC{0}".format(x + 1) for x in range(summary.shape[0])]

    # Convert loadings, scores and summaries to Pandas DataFrame rename index
    df_scores = pd.DataFrame(data=scores, index=data.index, columns=header)
    df_loadings = pd.DataFrame(data=loadings, index=header, columns=data.columns)
    df_summary = pd.DataFrame(data=summary, index=header, columns=["standard_deviation",
                                                                   "proportion_of_variance_explained",
                                                                   "cumulative_proportion_of_variance_explained"])

    return df_scores, df_loadings, df_summary

def slice_and_PCA(pandas_data, column_name, group_name):

    #Slice numeric data
    num_data = pandas_data[pandas_data[column_name] == group_name].select_dtypes(include=[np.number])

    #Standardize and run PCA
    std_data = pd.DataFrame(StandardScaler().fit_transform(num_data))
    scores_data, loadings_data, summary_data = runPCA(std_data)
    return scores_data

def iso(std_data, num_neighbors, num_comp):
    '''
    Takes standardized pandas_data and returns ISOMAP component pandas list
    :param std_data: A pandas array containing standardized data
    :param num_neighbors: number of neighbors to use in ISOMAP
    :param num_comp: number of components to be written to output pandas array
    :return: Pandas array with each column containing Component and its values
    '''
    iso = manifold.Isomap(n_neighbors=num_neighbors, n_components=num_comp)
    iso.fit(std_data)
    pre_data = iso.transform(std_data)
    col_list = ["Component {}".format(i) for i in range(1, num_comp+1)]
    return pd.DataFrame(pre_data, columns=col_list)

def tsne_reduction(pandas_data, n_comps=2, verb=1, perp=30, num_iter=300, initial="random"):
    '''
    Returns a pandas dataframe containing the specified number of components generated by\
    t-SNE reduction
    :param pandas_data: standadrized dataset
    :param n_comps: Number of components to output into table
    :param verb: Verbosity level of t-SNE
    :param perp: Perplexity, recommended range is between 5-50
    :param num_iter: Number of iterations to perform
    :param initial: How to initialize data, given as either "random" or "pca"
    :return: Pandas dataframe with calculated components
    '''
    tsne = TSNE(n_components = n_comps, verbose = verb, perplexity = perp, n_iter=num_iter, init=initial)
    tsne_results = tsne.fit_transform(pandas_data.values)
    col_list = ["Component {}".format(i) for i in range(1, n_comps+1)]
    return pd.DataFrame(tsne_results, columns=col_list)