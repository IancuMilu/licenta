import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.stats import zscore

# Perform Gaussian Mixture clustering on a dataset and plot the silhouette scores for each cluster.
def gmm_silhouette_clustering(file_name, feature1, feature2, cluster_number, feature1_name, feature2_name, process_id=None):
    # Read in CSV file and extract specified features
    selected_headers = [feature1, feature2, 'Class']
    data = pd.read_csv(file_name, usecols=selected_headers)

    # Group the data by the driver column
    grouped = data.groupby('Class')
    num_drivers = len(grouped)

    # Calculate the number of rows and columns for subplots
    num_rows = 2
    num_cols = 2

    # Create a figure for Gaussian Mixture clusters
    fig_gmm, axs_gmm = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 10))
    fig_gmm.suptitle('Silhouette Analysis for Gaussian Mixture Clustering', fontsize=16)
    fig_gmm.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Initialize lists to store KMeans and Gaussian Mixture scores for each driver
    gmm_scores = []
    avg_silhouette_scores = []

    # Define custom cluster labels and number of clusters
    cluster_labels = ['Cluster 1', 'Cluster 2'] 
    cluster_colors = ['blue', 'red']
    cluster_number = 2

    # Loop through each driver group
    for i, (name, group) in enumerate(grouped):
        # Remove outliers using z-score method
        group = group[(np.abs(zscore(group.iloc[:,1:])) < 3).all(axis=1)]

        # Standardize the data
        scaler = StandardScaler()
        group_std = scaler.fit_transform(group.iloc[:,1:])

        # Create Gaussian Mixture model and fit to data
        gmm = GaussianMixture(n_components=cluster_number, random_state=0)
        gmm.fit(group_std)

        # Calculate Gaussian Mixture silhouette samples
        gmm_labels = gmm.predict(group_std)
        gmm_silhouette_samples = silhouette_samples(group_std, gmm_labels)

        # Calculate Gaussian Mixture silhouette score
        gmm_score = silhouette_score(group_std, gmm_labels)
        gmm_scores.append(gmm_score)    

        # Create Gaussian Mixture silhouette graph
        y_lower = 0
        for j in range(cluster_number):
            ith_cluster_silhouette_values = gmm_silhouette_samples[gmm_labels == j]
            ith_cluster_silhouette_values.sort()
            size_cluster_j = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j
            color = cluster_colors[j]
            axs_gmm[i // num_cols, i % num_cols].fill_betweenx(np.arange(y_lower, y_upper), 0,
                                                ith_cluster_silhouette_values, facecolor=color,
                                                edgecolor=color, alpha=0.7)
            cluster_label = cluster_labels[j]
            axs_gmm[i // num_cols, i % num_cols].text(-0.05, y_lower + 0.5 * size_cluster_j, cluster_label)
            y_lower = y_upper + 10

        # Add x-label and y-label
        axs_gmm[i // num_cols, i % num_cols].set_xlabel('Silhouette Coefficient Values')
        axs_gmm[i // num_cols, i % num_cols].set_ylabel('Samples within cluster')

        # Calculate average silhouette score for the Gaussian Mixture model
        avg_silhouette_score = np.mean(gmm_silhouette_samples)
        avg_silhouette_scores.append(avg_silhouette_score)

        # Print the average silhouette score for the cluster
        print(f"Driver {name}, Avg. Silhouette Score: {avg_silhouette_score:.2f}")

        # Add vertical line for average silhouette score
        axs_gmm[i // num_cols, i % num_cols].axvline(x=avg_silhouette_score, color='red', linestyle='--')
        axs_gmm[i // num_cols, i % num_cols].set_title(f'Driver {name}')

        # Break the loop if we have plotted the desired number of drivers
        if i+1 == num_rows * num_cols:
            break

    # Save the plot
    if process_id is not None:
        print(f"Process {process_id} completed Gaussian Mixture Model silhouette clustering.")
    fig_gmm.set_size_inches(10, 10)
    filename = f'Fig/gmm_silhouette_{feature1_name}_{feature2_name}.png'
    basename, extension = os.path.splitext(filename)
    i = 1
    while os.path.isfile(filename):
        filename = f'{basename}_{i}{extension}'
        i += 1
    fig_gmm.savefig(filename, dpi=100)
    print(f"Gaussian Mixture Model Silhouette figure saved as gmm_silhouette_{feature1_name}_{feature2_name}.png")