import pandas as pd
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from ourStatistics import interpolate_by_mean as ibm, runPCA

#Read data and interpolate values
cortex_data = pd.read_excel('Data_Cortex_Nuclear.xls').apply(ibm, axis=0)

#Slice data and take only numeric data
cCSs_data = cortex_data[cortex_data['class'] == 'c-SC-s'].select_dtypes(include=[np.number])
tCSs_data = cortex_data[cortex_data['class'] == 't-SC-s'].select_dtypes(include=[np.number])

#Standradize data for PCA
cCSs_data = pd.DataFrame(StandardScaler().fit_transform(cCSs_data))
tCSs_data = pd.DataFrame(StandardScaler().fit_transform(tCSs_data))

#Run PCA
scores_cCSs, loadings_cCSs, summary_cCSs = runPCA(cCSs_data)
scores_tCSs, loadings_tCSs, summary_tCSs = runPCA(tCSs_data)

#Turn data into ISOMAP
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

cCSs_iso = iso(cCSs_data, 10, 2)
tCSs_iso = iso(tCSs_data, 10, 2)

#Plot the 2 side-by-side for comparison
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].set_title("Principal Component Analysis")
ax[0].scatter(scores_cCSs["PC1"], scores_cCSs["PC2"], color='blue', label="control")
ax[0].scatter(scores_tCSs["PC1"], scores_tCSs["PC2"], color='red', label="treatment")

ax[1].set_title("ISOMAP")
ax[1].scatter(cCSs_iso['Component 1'], cCSs_iso['Component 2'], color='blue', label="control")
ax[1].scatter(tCSs_iso['Component 1'], tCSs_iso['Component 2'], color='red', label="treatment")

ax[0].legend(loc="upper left")

fig.savefig('dimension reduction comparison.png')
fig.show()
