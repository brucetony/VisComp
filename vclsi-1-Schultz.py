import pandas as pd
import numpy as np

# Part a
bc_data = pd.read_excel('breast-cancer-wisconsin.xlsx')
print("Number of instances:", bc_data.shape[0],  "\nNumber of columns:", bc_data.shape[1],  "\nList of column names:", bc_data.columns.values)

# Part c
benign = bc_data[bc_data['class'] == 2]
malig = bc_data[bc_data['class'] == 4]
print("Benign instances:", benign.shape[0], "\nMalignant instances:", malig.shape[0])

#Part d
def F_score(attribute, group1, group2):
    '''
    Calculates the F score between two groups using a common attribute
    :param attribute: Shared attribute of interest
    :param group1: Subgroup of a dataframe, must be pandas dataframe
    :param group2: Another pandas dataframe that is a subgroup of the same dataset as group1
    :return: F score for the attribute of group1 and group2
    '''
    if not isinstance(group1, pd.core.frame.DataFrame) or not isinstance(group2, pd.core.frame.DataFrame):
        raise TypeError("Both groups need to be a pandas dataframe!")

    combined_data = pd.concat([group1, group2]) #Used to generate mean for all data

    grand_mean = np.mean(combined_data[attribute])
    g1_mean = np.mean(group1[attribute])
    g2_mean = np.mean(group2[attribute])

    g1_diff = [(i-g1_mean)**2 for i in group1[attribute]]
    g2_diff = [(i-g2_mean)**2 for i in group2[attribute]]

    numerator = (g1_mean - grand_mean)**2 + (g2_mean - grand_mean)**2
    denom = (1/(len(group1)-1))*sum(g1_diff) + (1/(len(group2)-1))*sum(g2_diff)

    F = numerator/denom

    return F


print(benign.isnull().values.any())
print(F_score('marAdh', benign, malig))