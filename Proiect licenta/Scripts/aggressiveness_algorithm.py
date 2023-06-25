import pandas as pd

# Read the CSV file into a Pandas dataframe
df = pd.read_csv("dataset.csv")

# Define the column names to modify
cols_to_modify = ['Engine_speed', 'Vehicle_speed']

# Group the dataframe by 'Class' and iterate over each group
for driver_name, group in df.groupby('Class'):
    # Get the rows for this driver and the columns to modify
    driver_rows = group.loc[:, cols_to_modify]

    # Calculate the averages of the columns to modify for this driver
    avg_cols = driver_rows.mean()

    print(f"Driver: {driver_name}")
    print("Averages:")
    print(avg_cols)
    print("----------------------")

    skip_counter = 0
    modification_counter = 0
    for i, row in driver_rows.iterrows():
        if skip_counter == 20 and modification_counter > 0:
            modification_counter -= 1
            if modification_counter == 0:
                skip_counter == 0
            continue        
            
        else:
            if (row[cols_to_modify] > avg_cols).all():
                driver_rows.loc[i] *= 1.5
                skip_counter += 1
                modification_counter += 1

    # Update the modified rows in the original dataframe
    df.update(driver_rows)

# Write the updated dataframe to a new csv file with 'mod_' prefix
new_file_name = 'output.csv'
df.to_csv(new_file_name, index=False)
