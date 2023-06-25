import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

def kmeans_clustering(file_name, feature1, feature2, cluster_number, feature1_name, feature2_name, process_id=None):
    # Read in CSV file and extract specified features
    selected_headers = [feature1, feature2, 'Class']
    data = pd.read_csv(file_name, usecols=selected_headers)

    # Group the data by the driver column
    grouped = data.groupby('Class')

    # Create a figure for KMeans clusters
    fig_kmeans, axs_kmeans = plt.subplots(nrows=2, ncols=5, figsize=(15, 20))
    fig_kmeans.suptitle('Kmeans Clusters', fontsize=16)
    fig_kmeans.subplots_adjust(wspace=0.2, hspace=0.3)

    # Loop through each driver group
    for i, (name, group) in enumerate(grouped):

        # Remove outliers using z-score method
        group = group[(np.abs(zscore(group.iloc[:,1:])) < 3).all(axis=1)]

        # Standardize the data
        scaler = StandardScaler()
        group_std = scaler.fit_transform(group.iloc[:,1:])

        # Create KMeans model and fit to data
        kmeans = KMeans(n_clusters=cluster_number, random_state=0, n_init=10)
        kmeans.fit(group_std)

        # Check for smaller centroid and change label or color
        if kmeans.cluster_centers_[0].sum() < kmeans.cluster_centers_[1].sum():
            cluster_labels = ['Cluster 1', 'Cluster 2']
            cluster_colors = ['blue', 'red']
        else:
            cluster_labels = ['Cluster 2', 'Cluster 1']
            cluster_colors = ['red', 'blue']            

        # Create KMeans cluster plot
        kmeans_handles = axs_kmeans[i // 5, i % 5].scatter(group[feature1], group[feature2], c=kmeans.labels_,
                                                      cmap=ListedColormap(cluster_colors))
        axs_kmeans[i // 5, i % 5].set_title(f'Driver {name}')
        axs_kmeans[i // 5, i % 5].set_xlabel(feature1_name)
        axs_kmeans[i // 5, i % 5].set_ylabel(feature2_name)
        axs_kmeans[i // 5, i % 5].legend(handles=kmeans_handles.legend_elements()[0], labels=cluster_labels)

    # Save the plot
    if process_id is not None:
        print(f"Process {process_id} completed k-means clustering.")
    fig_kmeans.set_size_inches(22, 11)
    filename = f'Fig/kmeans_clusters_{feature1_name}_{feature2_name}.png'
    basename, extension = os.path.splitext(filename)
    i = 1
    while os.path.isfile(filename):
        filename = f'{basename}_{i}{extension}'
        i += 1
    fig_kmeans.savefig(filename, dpi=100)
    print(f"KMeans clusters figure saved as {filename}")
