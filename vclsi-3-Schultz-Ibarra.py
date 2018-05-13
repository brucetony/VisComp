import pandas as pd
import numpy as np
from scipy import stats
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from ourStatistics import interpolate_by_mean as ibm, runPCA


def iso(std_data, num_neighbors, num_comp):
    """
    Takes standardized pandas_data and returns ISOMAP component pandas list
    :param std_data: A pandas array containing standardized data
    :param num_neighbors: number of neighbors to use in ISOMAP
    :param num_comp: number of components to be written to output pandas array
    :return: Pandas array with each column containing Component and its values
    """
    iso = manifold.Isomap(n_neighbors=num_neighbors, n_components=num_comp)
    iso.fit(std_data)
    pre_data = iso.transform(std_data)
    col_list = ["Component {}".format(i) for i in range(1, num_comp + 1)]
    return pd.DataFrame(pre_data, columns=col_list)


# Read data and interpolate values
cortex_data = pd.read_excel('Data_Cortex_Nuclear.xls').apply(ibm, axis=0)

# Slice data and take only numeric data
cCSs_data = cortex_data[cortex_data['class'] == 'c-SC-s'].select_dtypes(include=[np.number])
tCSs_data = cortex_data[cortex_data['class'] == 't-SC-s'].select_dtypes(include=[np.number])

# Standradize data for PCA
cCSs_data = pd.DataFrame(StandardScaler().fit_transform(cCSs_data))
tCSs_data = pd.DataFrame(StandardScaler().fit_transform(tCSs_data))

# Run PCA
scores_cCSs, loadings_cCSs, summary_cCSs = runPCA(cCSs_data)
scores_tCSs, loadings_tCSs, summary_tCSs = runPCA(tCSs_data)

# Get ISOMAP scores
cCSs_iso = iso(cCSs_data, 10, 2)
tCSs_iso = iso(tCSs_data, 10, 2)

Plot the 2 side-by-side for comparison
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

# Part c

# Read breast cancer data and interpolate values
bc_data = pd.read_excel('breast-cancer-wisconsin.xlsx').apply(ibm, axis=0)

# Standardize dataset - except class
class_col = bc_data.pop('class')
# bc_data_std = pd.DataFrame(StandardScaler().fit_transform(bc_data))  # Standardize
bc_data_std = pd.DataFrame(MinMaxScaler().fit_transform(bc_data))  # Normalize
bc_data_std['class'] = class_col


# Define a simple t-SNE function
def tsne_reduction(pandas_data, n_comps=2, verb=1, perp=30, num_iter=300, initial="random"):
    """
    Returns a pandas data frame containing the specified number of components generated by
    t-SNE reduction
    :param pandas_data: standardized data set
    :param n_comps: Number of components to output into table
    :param verb: Verbosity level of t-SNE
    :param perp: Perplexity, recommended range is between 5-50
    :param num_iter: Number of iterations to perform
    :param initial: How to initialize data, given as either "random" or "pca"
    :return: Pandas data frame with calculated components
    """
    tsne = TSNE(n_components=n_comps, verbose=verb, perplexity=perp, n_iter=num_iter, init=initial)
    tsne_results = tsne.fit_transform(pandas_data.values)
    col_list = ["Component {}".format(i) for i in range(1, n_comps + 1)]
    return pd.DataFrame(tsne_results, columns=col_list)

def remove_outliers(pandas_data, within_std=3):
    '''
    Calculates z score for pandas dataframe and returns dataframe with entries that are within chosen deviation away
    :param pandas_data: Pandas Dataframe of data
    :param within_std: Number of standard deviations away from mean to accept
    :return: Dataframe with rows removed that are outside specified range
    '''
    return pandas_data[(np.abs(stats.zscore(pandas_data)) < within_std).all(axis=1)]


# Create lists of perplexities and initializing schemes to test
perplexities = [5, 10, 20, 30, 40, 50]
initiate = ['random', 'pca']

# Generate new pandas table for our results
bc_data_tsne = bc_data_std.copy()
colors = ["blue" if row is 2 else "red" for row in list(bc_data_tsne['class'])]
labels = ["benign" if row is 2 else "malignant" for row in list(bc_data_tsne['class'])]
bc_data_tsne['colors'] = colors
bc_data_tsne['labels'] = labels

# Iterate over perplexities and generate subplots
for j in range(len(initiate)):
    # Create initial figure and set legend/titles
    fig, ax = plt.subplots(1, len(perplexities), figsize=(30, 5))
    legend_ele = [Line2D([0], [0], color='w', label='benign', marker='o', markerfacecolor='b'),\
                  Line2D([0], [0], color='w', label='malignant', marker='o', markerfacecolor='r')]
    ax[0].legend(handles=legend_ele, loc='upper left')
    fig.suptitle("{} initialization".format(initiate[j]), x=0.5, y=1.0, fontsize=18)
    for i in range(len(perplexities)):
        tsne_comps = tsne_reduction(bc_data, perp=i, initial=initiate[j], num_iter=300)

        # Remove outliers to allow us to see clusters/trends
        #tsne_comps = remove_outliers(tsne_comps, within_std=3) # Uses zscore
        tsne_comps = tsne_comps[tsne_comps.apply(lambda x: np.abs(x - x.median()) / x.std() < 3).all(axis=1)] # Median

        # Add data to our tsne dataframe
        bc_data_tsne['{}_Perp{}_Component 1'.format(initiate[j], perplexities[i])] = tsne_comps['Component 1']
        bc_data_tsne['{}_Perp{}_Component 2'.format(initiate[j], perplexities[i])] = tsne_comps['Component 2']

        # Create graphs
        ax[i].set_title("Perplexity = {}".format(perplexities[i]))
        ax[i].scatter(bc_data_tsne['{}_Perp{}_Component 1'.format(initiate[j], perplexities[i])], \
                      bc_data_tsne['{}_Perp{}_Component 2'.format(initiate[j], perplexities[i])] \
                      , color=colors, label=labels)
        ax[i].set_xlabel("{} Component 1".format(initiate[j]))
        if i == 0:
            ax[i].set_ylabel("{} Component 2".format(initiate[j]))

    plt.show()

