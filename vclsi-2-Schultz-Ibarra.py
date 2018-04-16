#Exercise 1

import matplotlib.pyplot as plt
import pandas as pd

bc_data = pd.read_csv('reduced_data.csv')

scat_matrix = pd.plotting.scatter_matrix(bc_data, figsize=(5, 5), diagonal='hist')
scat_matrix.colors=['red', 'green']

plt.show()

head(bc_data)
