#Exercise 1

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

bc_data_full = pd.read_excel('breast-cancer-wisconsin.xlsx')

def color_list(variableList):
    color_list = []
    for row in variableList:
        if row == 2:
            color_list.append('blue')
        else:
            color_list.append('red')
    return color_list

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
    #TODO check if this is right
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


def scat_matrix():
    #TODO make more general, input parameters = data + list of columns + group separator?
    # Using pandas just to import data
    columns = list(pd.read_csv('reduced_data.csv').columns.values)
    if columns[0] == 'Unnamed: 0': #In case using old data set
        columns = columns[1:]
    columns.remove('bareNuc')  # Temporary because of NaN, have to deal with those
    bc_data = pd.read_excel('breast-cancer-wisconsin.xlsx')

    # Generate figure for plots to be plotted in
    size = len(columns)
    fig, subs = plt.subplots(size, size)

    #Create colors list
    colors = color_list(list(bc_data['class']))

    #Used for tracking plot placement
    i = 0
    j = 0

    #Iterate through column names and generate plots
    for axis1 in columns:
        for axis2 in columns:
            if axis1 == axis2: #Make diagonal plots different
                #TODO Interpolate missing values
                benign_sd = list(bc_data[bc_data['class'] == 2][axis1])
                malig_sd = list(bc_data[bc_data['class'] == 4][axis1])

                #Generate histogram, density parameter means normalized
                subs[i, j].hist(benign_sd, color = 'blue', label = 'benign', histtype='bar', density=True)
                subs[i, j].hist(malig_sd, color='red', label='malig', histtype='bar', density=True)
                subs[i, j].set_title(axis1) #Can you optional 'y=###' parameter to move title
                #TODO set axex ranges
                #axes[i, j].xlabel(axis1)
                #axes[i, j].ylabel('Frequency')
                #TODO a third thing is being plotted as well --> FIND AND REMOVE
                #TODO put border around bar
                #TODO flip so diagonal goes the other way?
                #TODO verify why some graphs look like >1 when normalized
            else:
                #Set low alpha so overlapping points look darker, alter size of dot instead?
                subs[i, j].scatter(bc_data[axis1], bc_data[axis2], alpha=0.1, c=colors, s=2)
                benign_a1 = list(bc_data[bc_data['class'] == 2][axis1])
                malig_a1 = list(bc_data[bc_data['class'] == 4][axis1])
                benign_a2 = list(bc_data[bc_data['class'] == 2][axis2])
                malig_a2 = list(bc_data[bc_data['class'] == 4][axis2])
                all_points = [list(zip(benign_a1, benign_a2)), list(zip(malig_a1, malig_a2))]
                print('DSC of {} and {} = {}'.format(axis1, axis2, DSC(all_points)))

            # Increase counter
            if i < size-1:
                i += 1
            else:
                i = 0
                j += 1

    # Figure attributes
    subs[0, 0].legend(bbox_to_anchor=(-0.2, 1.0))
    fig.suptitle('Malignant and Benign Tumor Values', x=0.5, y=1.0)
    fig.show()

scat_matrix()
