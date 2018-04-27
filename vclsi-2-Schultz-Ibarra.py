#Exercise 1

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

bc_data_full = pd.read_excel('breast-cancer-wisconsin.xlsx')

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
    # TODO check if this is right
    # First computer center points
    centers = cluster_center(point_list)
    # Create empty lists to store distances of the difference between each point and each centroid
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

def find_missing_values(pds_array):
    '''
    :param A pandas dataframe:
    :return: A list containing the column names which have null values
    '''
    null_vectors = []
    for i in pds_array.columns.values:
        if pds_array[i].isnull().values.any():
            null_vectors.append(i) #Add to list of vectors with missing values
            print("{} contains null values...fixing".format(i))
    if not null_vectors:
        print("No null values were found")
    return null_vectors

def interpolate_values(pandasDF):
    '''
    Interpolate null values using average for vector by class
    :param pandasDF: Full pandas dataframe
    :return: Concatenated pandas dataframe with null values replaced
    '''
    null_vectors = find_missing_values(pandasDF)
    for vector in null_vectors:
        # Slice empty vectors
        benign = pandasDF[pandasDF['class'] == 2]
        malig = pandasDF[pandasDF['class'] == 4]

        # Drop na and find average for each class, rounded to nearest integer
        benign_av = int(round(np.mean(benign[vector].dropna())))
        malig_av = int(round(np.mean(malig[vector].dropna())))

        # Fill null with average values
        benign = benign.fillna(benign_av)
        malig = malig.fillna(malig_av)

        # Concatenate pandas frames
        pandasDF_no_null = pd.concat([benign, malig])
    return pandasDF_no_null


def scat_matrix():
    #TODO make more general, input parameters = data + list of columns + group separator?
    # Using pandas just to import data
    columns = list(pd.read_csv('reduced_data.csv').columns.values)
    # In case using old data set
    if columns[0] == 'Unnamed: 0':
        columns = columns[1:]
    bc_data = pd.read_excel('breast-cancer-wisconsin.xlsx')

    # Interpolate missing data
    bc_data = interpolate_values(bc_data)

    # Generate figure for plots to be plotted in
    size = len(columns)
    fig, subs = plt.subplots(size, size, figsize=(15, 15))

    # Used for tracking plot placement
    i = 0
    j = 0

    # Create colors list
    colors = ["blue" if row is 2 else "red" for row in list(bc_data['class'])]

    # Separate Data
    benign_data = bc_data[bc_data['class'] == 2]
    malig_data = bc_data[bc_data['class'] == 4]

    # Collect DSC values
    DSC_values = []

    # Iterate through column names and generate plots
    for axis1 in columns:
        for axis2 in columns:
            if axis1 == axis2:
                # Make diagonal plots different
                benign_sd = list(benign_data[axis1])
                malig_sd = list(malig_data[axis1])

                # Generate histogram, density parameter means normalized
                subs[i, j].hist(benign_sd, color = 'blue', label = 'benign', histtype='bar', edgecolor='black')
                subs[i, j].hist(malig_sd, color='red', label='malig', histtype='bar', edgecolor='black')
                subs[i, j].set_title(axis1)
                #axes[i, j].xlabel(axis1)
                #axes[i, j].ylabel('Frequency')
            else:
                # Generate sizes by counting number of times a point occurs
                datapoints = list(zip(bc_data[axis1], bc_data[axis2]))
                sizes = [datapoints.count(point)*2 for point in datapoints]

                # Create scatter plot at position i,j using created sizes/colors, alpha used for overlapping points
                subs[i, j].scatter(bc_data[axis1], bc_data[axis2], alpha=0.25, c=colors, s=sizes)

                # Setup to calculate DSC
                benign_scatter = list(zip(benign_data[axis1], benign_data[axis2]))
                malig_scatter = list(zip(malig_data[axis1], malig_data[axis2]))
                all_points = [benign_scatter, malig_scatter]
                DSC_values.append(('{} and {}'.format(axis1, axis2), DSC(all_points)))

            # Increase counter to change position
            if i < size-1:
                i += 1
            else:
                i = 0
                j += 1

    # Figure attributes
    subs[0, 0].legend(bbox_to_anchor=(-0.2, 1.0))
    fig.suptitle('Malignant and Benign Tumor Values', x=0.5, y=1.0)
    #fig.savefig('Scatter Plot.png')
    fig.show()

    # Determine max DSC value and return highest one
    DSC_values.sort(key=lambda x: x[1])
    print("The highest distance consistancy is {} and found between {}".format(DSC_values[-1][1], DSC_values[-1][0]))


scat_matrix()