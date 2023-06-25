import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

def gaussian_mixture_clustering(file_name, feature1, feature2, num_clusters, feature1_name, feature2_name, process_id=None):
    # Read in CSV file and extract specified features
    selected_headers = [feature1, feature2, 'Class']
    data = pd.read_csv(file_name, usecols=selected_headers)

    # Group the data by the driver column
    grouped = data.groupby('Class')

    # Create a figure for Gaussian Mixture Model clusters
    fig_gmm, axs_gmm = plt.subplots(nrows=2, ncols=5, figsize=(15, 20))
    fig_gmm.suptitle('Gaussian Mixture Model Clusters', fontsize=16)
    fig_gmm.subplots_adjust(wspace=0.2, hspace=0.3)

    # Loop through each driver group
    for i, (name, group) in enumerate(grouped):

        # Remove outliers using z-score method
        group = group[(np.abs(zscore(group.iloc[:,1:])) < 3).all(axis=1)]

        # Standardize the data
        scaler = StandardScaler()
        group_std = scaler.fit_transform(group.iloc[:,1:])

        # Create Gaussian Mixture Model and fit to data
        gmm = GaussianMixture(n_components=num_clusters, random_state=0)
        gmm.fit(group_std)

        # Check for smaller centroid and change label or color
        if gmm.means_[0].sum() < gmm.means_[1].sum():
            cluster_labels = ['Cluster 1', 'Cluster 2']
            cluster_colors = ['blue', 'red']
        else:
            cluster_labels = ['Cluster 2', 'Cluster 1']
            cluster_colors = ['red', 'blue']

        # Create Gaussian Mixture Model cluster plot
        gmm_handles = axs_gmm[i // 5, i % 5].scatter(group[feature1], group[feature2], c=gmm.predict(group_std),
                                                cmap=ListedColormap(cluster_colors))
        axs_gmm[i // 5, i % 5].set_title(f'Driver {name}')
        axs_gmm[i // 5, i % 5].set_xlabel(feature1_name)
        axs_gmm[i // 5, i % 5].set_ylabel(feature2_name)
        axs_gmm[i // 5, i % 5].legend(handles=gmm_handles.legend_elements()[0], labels=cluster_labels)

    # Save the plot
    if process_id is not None:
        print(f"Process {process_id} completed gmm clustering.")
    fig_gmm.set_size_inches(22, 11)
    filename = f'Fig/gmm_clusters_{feature1_name}_{feature2_name}.png'
    basename, extension = os.path.splitext(filename)
    i = 1
    while os.path.isfile(filename):
        filename = f'{basename}_{i}{extension}'
        i += 1
    fig_gmm.savefig(filename, dpi=100)
    print(f"GMM clusters figure saved as {filename}")
