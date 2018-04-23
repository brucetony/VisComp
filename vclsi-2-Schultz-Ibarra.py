#Exercise 1

import matplotlib.pyplot as plt
import pandas as pd

bc_data_full = pd.read_excel('breast-cancer-wisconsin.xlsx')

def color_list(variableList):
    color_list = []
    for row in variableList:
        if row == 2:
            color_list.append('blue')
        else:
            color_list.append('red')
    return color_list

def DC():

    pass

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
                #TODO  Determine how to drop NaN properly and put missing column back in
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

#scat_matrix()
#print((bc_data_full[bc_data_full['class'] == 2]['normNuc']))
