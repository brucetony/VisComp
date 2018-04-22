#Exercise 1

import matplotlib.pyplot as plt
import pandas as pd

#Using pandas just to import data
bc_data = pd.read_csv('reduced_data.csv')
bc_data = bc_data.drop(['bareNuc'], axis=1) #Temporary because of NaN, have to deal with those
bc_data_full = pd.read_excel('breast-cancer-wisconsin.xlsx')

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

            # benign_counts = {x:all_sd[0].count(x) for x in all_sd[0]}
            # malig_counts = {x:all_sd[1].count(x) for x in all_sd[1]}
            # benign_data_points = []
            # malig_data_points = []
            # for value, instance in benign_counts.items():
            #     benign_data_points.append((value, instance/len(list(benign_sd[axis1]))))
            # for value, instance in malig_counts.items():
            #     malig_data_points.append((value, instance / len(list(malig_sd[axis1]))))
            # benign_x = [x[0] for x in benign_data_points]
            # benign_y = [x[1] for x in benign_data_points]
            # malig_x = [x[0] for x in malig_data_points]
            # malig_y = [x[1] for x in malig_data_points]
            pass
        else:
            plt.scatter(bc_data[axis1], bc_data[axis2])
            #plt.show()
            pass
#print(bc_data.columns.values)
#print((bc_data_full[['uniCelShape', 'class']]))