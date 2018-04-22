#Exercise 1

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Using pandas just to import data
bc_data = pd.read_csv('reduced_data.csv')
bc_data = bc_data.drop(['bareNuc'], axis=1) #Temporary because of NaN, have to deal with those
bc_data_full = pd.read_excel('breast-cancer-wisconsin.xlsx')

def scat_matrix(pandasFile, columns, groupColumnName):
    # Generate figure for plots to be plotted in
    size = len(bc_data.columns.values)
    fig, axes = plt.subplots(size, size)

    i = 0
    j = 0
    for axis1 in bc_data.columns.values:
        for axis2 in bc_data.columns.values:
            if axis1 == axis2: #Make diagonal plots different
                select_data = bc_data_full[[axis1, 'class']]
                # TODO  Determine how to drop NaN properly
                #select_data = select_data.dropna(axis=1, how='any')
                benign_sd = select_data[select_data['class'] == 2].drop('class', axis=1)
                malig_sd = select_data[select_data['class'] == 4].drop('class', axis=1)
                # Group data in list
                all_sd = [list(benign_sd[axis1]), list(malig_sd[axis1])]
                #Generate histogram
                axes[i, j].hist(all_sd[0], color = 'blue', label = 'benign', histtype='bar', density=True)
                axes[i, j].hist(all_sd[1], color='red', label='malig', histtype='bar', density=True)
                axes[i, j].set_title(axis1) #Can you optional 'y=###' parameter to move title
                #TODO set axex ranges
                #axes[i, j].xlabel(axis1)
                #axes[i, j].ylabel('Frequency')
                #plt.legend(loc='upper right')
                #TODO a third thing is being plotted as well
                #TODO remove column names from legend
                #TODO put border around bar
                #TODO flip so diagonal goes the other way?

            else:
                axes[i, j].scatter(bc_data[axis1], bc_data[axis2], s=2)

            # Create counting tool to assemble subplots correctly
            if i < size-1:
                i += 1
            else:
                i = 0
                j += 1


    plt.show()

print((list(bc_data_full['class'])))