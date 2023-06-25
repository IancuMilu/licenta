import pandas as pd

# Read the CSV file
csv_file = 'dataset.csv'
data = pd.read_csv(csv_file)

# Filter numeric columns
numeric_columns = data.select_dtypes(include=[float, int]).columns

# Calculate the correlation matrix
correlation_matrix = data[numeric_columns].corr()

# Save correlation matrix to CSV
output_file_csv = 'correlation_matrix.csv'
correlation_matrix.to_csv(output_file_csv, index=True)

# Find pairs with correlation coefficient over 0.8
pairs_above_threshold = correlation_matrix.abs().stack()[(correlation_matrix.abs().stack() > 0.8) & (correlation_matrix.abs().stack() < 1)]
pairs = [(pair[0], pair[1], correlation_matrix.loc[pair[0], pair[1]]) for pair in pairs_above_threshold.index]

# Remove duplicate pairs and pairs with identical features
unique_pairs = set()
filtered_pairs = []
for pair in pairs:
    feature1, feature2, correlation_coefficient = pair
    if feature1 != feature2 and (feature2, feature1) not in unique_pairs:
        filtered_pairs.append((feature1, feature2, correlation_coefficient))
        unique_pairs.add(pair[:2])

# Sort pairs based on the correlation coefficient in descending order
sorted_pairs = sorted(filtered_pairs, key=lambda x: x[2], reverse=True)

# Save pairs to TXT file
output_file_txt = 'pairs_above_threshold.txt'
with open(output_file_txt, 'w') as file:
    file.write("Feature 1, Feature 2, Correlation Coefficient\n")
    for pair in sorted_pairs:
        correlation_coefficient = round(pair[2], 2)
        file.write(f"{pair[0]}, {pair[1]}, {correlation_coefficient}\n")

print(f"Correlation matrix saved to '{output_file_csv}'")
print(f"Pairs with correlation coefficient over 0.8 saved to '{output_file_txt}'")
