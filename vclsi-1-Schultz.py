import pandas as pd
import numpy as np

# PART A
bc_data = pd.read_excel('breast-cancer-wisconsin.xlsx')
print("Number of instances:", bc_data.shape[0],  "\nNumber of columns:", bc_data.shape[1],  "\nList of column names:", bc_data.columns.values)

# PART B
# Create function to tell us which vectors have null values
def find_missing_values(pds_array):
    '''
    :param A pandas dataframe:
    :return: A list containing the column names which have null values
    '''
    null_vectors = []
    for i in pds_array.columns.values:
        if pds_array[i].isnull().values.any():
            null_vectors.append(i) #Add to list of vectors with missing values
            print("{} contains null values".format(i))
    if not null_vectors:
        print("No null values were found")
    return null_vectors

null_columns = find_missing_values(bc_data) #Use above function to find columns with missing values
for vec in null_columns:
    bc_data[vec] = bc_data[vec].interpolate()
find_missing_values(bc_data) #Check again to be sure

# TODO Explain why we use this method

# PART C
benign = bc_data[bc_data['class'] == 2].drop('class', axis = 1) #Drop class after sorting by it (gives no useful info)
malig = bc_data[bc_data['class'] == 4].drop('class', axis = 1)
bc_data_no_class = bc_data.drop('class', axis = 1)
print("Benign instances:", len(benign), "\nMalignant instances:", len(malig))

# PART D
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

    combined_data = pd.concat([group1, group2]) #Used to generate mean for all data (grand mean)

    grand_mean = np.mean(combined_data[attribute])
    g1_mean = np.mean(group1[attribute])
    g2_mean = np.mean(group2[attribute])

    g1_diff = [(i-g1_mean)**2 for i in group1[attribute]]
    g2_diff = [(i-g2_mean)**2 for i in group2[attribute]]

    numerator = (g1_mean - grand_mean)**2 + (g2_mean - grand_mean)**2
    denom = (1/(len(group1)-1))*sum(g1_diff) + (1/(len(group2)-1))*sum(g2_diff)

    F = numerator/denom

    return F

#Generate score for each attribute
scores = [F_score(attr, benign, malig) for attr in bc_data_no_class.columns.values]

#Now match attr with F score
matched = pd.DataFrame(
    {'Attribute': bc_data_no_class.columns.values, 'F score': scores}
)

#Sort scores
sorted = matched.sort_values(by = 'F score', ascending = False) #Sort scores/attr with largest at top

#Print top 5 attr with highest F scores (largest differences)
top_5 = sorted[0:5]
print(top_5)

#F value for 'class' will be infinity since you will divide by 0. The value will always match the mean since the value \
# for this particular attribute is uniform within each group

# PART E
# Take columns from the top 5 list and write reduced dataset to disk
reduced = bc_data[list(top_5['Attribute'])]
reduced.to_csv('reduced_data.csv') #Writes the index still to file

#TODO test datasets with online F score tool to verify function is working properly

