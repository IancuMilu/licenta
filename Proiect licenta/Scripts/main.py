import pandas as pd
import argparse
import multiprocessing as mp
import kmeans
import gmm
import kmeans_silhouette
import gmm_silhouette

# Define the command-line arguments
parser = argparse.ArgumentParser(description='Perform clustering on a CSV file.')
parser.add_argument('file_name', help='The CSV file to read data from')
parser.add_argument('feature1', type=str, help='name of the first feature to use')
parser.add_argument('feature2', type=str, help='name of the second feature to use')
parser.add_argument('num_clusters', type=int, help='number of clusters')
parser.add_argument('--kmeans', action='store_true', help='Calculate kmeans')
parser.add_argument('--gmm', action='store_true', help='Calculate gmm')
parser.add_argument('--kmeans_silhouette', action='store_true', help='Calculate silhouette score using KMeans clustering')
parser.add_argument('--gmm_silhouette', action='store_true', help='Calculate silhouette score using Gaussian Mixture clustering')

# Parse arguments
args = parser.parse_args()

# Read in CSV file and extract specified features
column_headers = pd.read_csv(args.file_name, nrows=1).columns.tolist()
if args.feature1 not in column_headers:
    print(f"{args.feature1} not found in CSV file.")
    exit()
if args.feature2 not in column_headers:
    print(f"{args.feature2} not found in CSV file.")
    exit()

# Create a list of functions to be called
functions = []
if args.kmeans:
    functions.append(kmeans.kmeans_clustering)
if args.gmm:
    functions.append(gmm.gaussian_mixture_clustering)
if args.kmeans_silhouette:
    functions.append(kmeans_silhouette.kmeans_silhouette_clustering)
if args.gmm_silhouette:
    functions.append(gmm_silhouette.gmm_silhouette_clustering)

if __name__ == '__main__':
    # Call each function using a separate process
    processes = []
    for i, func in enumerate(functions):
        p = mp.Process(target=func, args=(args.file_name, args.feature1, args.feature2, args.num_clusters, args.feature1, args.feature2, i))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()
