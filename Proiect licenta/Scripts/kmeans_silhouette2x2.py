import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.stats import zscore

# Perform KMeans clustering on a dataset and plot the silhouette scores for each cluster.
def kmeans_silhouette_clustering(file_name, feature1, feature2, cluster_number, feature1_name, feature2_name, process_id=None):
    # Read in CSV file and extract specified features
    selected_headers = [feature1, feature2, 'Class']
    data = pd.read_csv(file_name, usecols=selected_headers)

    # Group the data by the driver column
    grouped = data.groupby('Class')

    # Create a figure for KMeans clusters
    fig_kmeans, axs_kmeans = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig_kmeans.suptitle('Silhouette Analysis for KMeans Clustering', fontsize=16)
    fig_kmeans.subplots_adjust(wspace=0.3)
    for i, ax in enumerate(axs_kmeans.flat):
        ax.set(xlabel='Silhouette Coefficient Values', ylabel='Samples within cluster')
        ax.set_title(f'Driver {i+1}')

    # Initialize lists to store KMeans and Gaussian Mixture scores for each driver
    kmeans_scores = []
    avg_silhouette_scores = []

    # Define custom cluster labels and number of clusters
    cluster_labels = ['Cluster 1', 'Cluster 2'] 
    cluster_colors = ['blue', 'red']
    #cluster_number = 2

    # Loop through each driver group
    for i, (name, group) in enumerate(grouped):
        if i >= 4:
            break

        # Remove outliers using z-score method
        group = group[(np.abs(zscore(group.iloc[:,1:])) < 3).all(axis=1)]

        # Standardize the data
        scaler = StandardScaler()
        group_std = scaler.fit_transform(group.iloc[:,1:])

        # Create KMeans model and fit to data
        kmeans = KMeans(n_clusters=cluster_number, random_state=0, n_init=10)
        kmeans.fit(group_std)

        # Calculate KMeans silhouette score
        kmeans_score = silhouette_score(group_std, kmeans.labels_)
        kmeans_scores.append(kmeans_score)
        
        # Calculate KMeans silhouette samples
        kmeans_silhouette_samples = silhouette_samples(group_std, kmeans.labels_)

        # Create KMeans silhouette graph
        y_lower = 0
        for j in range(cluster_number):
            ith_cluster_silhouette_values = kmeans_silhouette_samples[kmeans.labels_ == j]
            ith_cluster_silhouette_values.sort()
            size_cluster_j = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j
            color = cluster_colors[j]
            axs_kmeans[i // 2, i % 2].fill_betweenx(np.arange(y_lower, y_upper), 0,
                                                    ith_cluster_silhouette_values, facecolor=color,
                                                    edgecolor=color, alpha=0.7)
            cluster_label = cluster_labels[j]
            axs_kmeans[i // 2, i % 2].text(-0.05, y_lower + 0.5 * size_cluster_j, cluster_label)
            y_lower = y_upper + 10

        # Calculate average silhouette score for the cluster
        avg_silhouette_score = np.mean(kmeans_silhouette_samples)
        avg_silhouette_scores.append(avg_silhouette_score)

        # Print the average silhouette score for the cluster
        print(f"Driver {name}, Avg. Silhouette Score: {avg_silhouette_score:.2f}")

        # Add vertical line for average silhouette score
        axs_kmeans[i // 2, i % 2].axvline(x=avg_silhouette_score, color='red', linestyle='--')
        axs_kmeans[i // 2, i % 2].set_title(f'Driver {name}')

    # Save the plot
    if process_id is not None:
        print(f"Process {process_id} completed KMeans silhouette clustering.")
    fig_kmeans.set_size_inches(12, 12)
    filename = f'Fig/kmeans_silhouette_{feature1_name}_{feature2_name}.png'
    basename, extension = os.path.splitext(filename)
    i = 1
    while os.path.isfile(filename):
        filename = f'{basename}_{i}{extension}'
        i += 1
    fig_kmeans.savefig(filename, dpi=100)
    print(f"KMeans silhouette figure saved as kmeans_silhouette_{feature1_name}_{feature2_name}.png")
