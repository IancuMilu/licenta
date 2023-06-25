import tkinter as tk
from tkinter import filedialog, messagebox
import re
import pandas as pd
import subprocess

selected_algorithms = []

def import_csv():
    # Open file dialog to select a CSV file
    file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])

    # Read the CSV file and create the DataFrame
    df = pd.read_csv(file_path)

    # Perform correlation calculation
    numeric_df = df.select_dtypes(include='number')
    global correlations
    correlations = numeric_df.corr()

    # Strip leading special characters from the header names
    headers = [re.sub(r'^[^a-zA-Z0-9_]+', '', header) for header in df.columns]

    # Exclude the header named "Class"
    headers = [header for header in headers if header != "Class"]

    # Update the dropdown menus with the headers
    dropdown1_menu['menu'].delete(0, 'end')
    dropdown2_menu['menu'].delete(0, 'end')
    for header in headers:
        dropdown1_menu['menu'].add_command(label=header, command=lambda h=header: update_dropdowns(h, dropdown2.get(), correlations))
        dropdown2_menu['menu'].add_command(label=header, command=lambda h=header: update_dropdowns(dropdown1.get(), h, correlations))

    # Store the header labels for updating dropdowns later
    global dropdown1_menu_headers
    global dropdown2_menu_headers
    dropdown1_menu_headers = headers.copy()
    dropdown2_menu_headers = headers.copy()

    # Set the selected values for the dropdown menus
    dropdown1.set("")
    dropdown2.set("")

    # Update the import label to indicate that the file was imported
    import_label.config(text="CSV file imported: " + file_path)

def update_dropdowns(header1, header2, correlations):
    dropdown1_menu['menu'].delete(0, 'end')
    dropdown2_menu['menu'].delete(0, 'end')
    for header in dropdown1_menu_headers:
        if header and header != header2:
            dropdown1_menu['menu'].add_command(label=header, command=lambda h=header: update_dropdowns(h, dropdown2.get(), correlations))
    for header in dropdown2_menu_headers:
        if header and header != header1:
            dropdown2_menu['menu'].add_command(label=header, command=lambda h=header: update_dropdowns(dropdown1.get(), h, correlations))

    # Set the selected values for the dropdown menus
    dropdown1.set(header1)
    dropdown2.set(header2)

    # Calculate correlation coefficient if both headers are selected
    if header1 and header2:
        correlation_coefficient = correlations.loc[header1, header2]
        correlation_coefficient = round(correlation_coefficient, 2)  # Round to two decimal places
        correlation_label.config(text=f"Correlation coefficient: {correlation_coefficient}")

        # Update the position of the correlation label dynamically based on the selected feature
        correlation_label.grid(row=dropdown2_menu.grid_info()["row"] + 1, column=dropdown2_menu.grid_info()["column"], sticky="w")
    else:
        correlation_label.config(text="")

def run_function():
    # Check if a CSV file is imported
    if import_label.cget("text") == "":
        messagebox.showwarning("CSV File Error", "Please import a CSV file before running the algorithms.")
        return
    
    # Get the selected values from the interface
    file_name = import_label.cget("text").replace("CSV file imported: ", "")
    feature1 = dropdown1.get()
    feature2 = dropdown2.get()
    num_clusters = 2

    # Check if features are selected
    if not feature1 or not feature2:
        messagebox.showwarning("Selection Error", "Please select both features to run the script.")
        return

    # Check if at least one algorithm is selected
    if not listbox.curselection():
        messagebox.showinfo("Error", "Please select at least one algorithm to run.")
        return

    # Create a list of selected algorithms based on listbox selections
    selected_algorithms = []
    for index in listbox.curselection():
        selected_algorithms.append(items[index])

    # Call the script and pass the arguments
    command = [
        "python",
        "scripts/main.py",
        file_name,
        feature1,
        feature2,
        str(num_clusters)
    ]

    # Add algorithm flags based on the selected algorithms
    for algorithm in selected_algorithms:
        if algorithm == "K-means":
            command.append("--kmeans")
        elif algorithm == "GMM":
            command.append("--gmm")
        elif algorithm == "K-means Silhouette":
            command.append("--kmeans_silhouette")
        elif algorithm == "GMM Silhouette":
            command.append("--gmm_silhouette")

    try:
        # Execute the command as a subprocess
        subprocess.run(command)
    finally:
        # Re-enable the window after the execution finishes
        window.grab_release()

    # Display a message and open the folder location
    messagebox.showinfo("Run Completed", "The script has finished running.")
    subprocess.run(["explorer", "Fig"])  # Replace "Fig" with the actual folder path
    subprocess.run(["explorer", "Results"])  # Replace "Fig" with the actual folder path


# Create the main window
window = tk.Tk()
window.title("Aplication")

# Calculate the width and height of the screen
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Set the desired width and height of the window
window_width = 650
window_height = 400

# Calculate the x and y coordinates to center the window
x = (window.winfo_screenwidth() - window_width) // 2
y = (window.winfo_screenheight() - window_height) // 2

# Set the geometry of the window
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Disable window resizing
window.resizable(False, False)

# Calculate the x and y coordinates to center the window
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2

# Set the geometry of the window
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Create the first dropdown menu
dropdown1 = tk.StringVar(window)
dropdown1.set("")  # Set initial value
dropdown1_label = tk.Label(window, text="Feature 1:")
dropdown1_label.grid(row=1, column=0, sticky="e", pady=10)
dropdown1_menu = tk.OptionMenu(window, dropdown1, "")
dropdown1_menu.configure(background="#6c7b8b", foreground="white")
dropdown1_menu.grid(row=1, column=1, sticky="w", pady=10)

# Create the second dropdown menu
dropdown2 = tk.StringVar(window)
dropdown2.set("")  # Set initial value
dropdown2_label = tk.Label(window, text="Feature 2:")
dropdown2_label.grid(row=2, column=0, sticky="e", pady=10)
dropdown2_menu = tk.OptionMenu(window, dropdown2, "")
dropdown2_menu.configure(background="#6c7b8b", foreground="white")
dropdown2_menu.grid(row=2, column=1, sticky="w", pady=10)

import_button = tk.Button(window, text="Import CSV", command=import_csv, bg="#6c7b8b", fg="white")
import_button.grid(row=0, column=0, sticky="w", pady=10, padx=10)

import_label = tk.Label(window, text="")
import_label.grid(row=0, column=1, sticky="w")

# Create the listbox for selecting items
listbox_label = tk.Label(window, text="Algorithms to use (using 2 clusters):")
listbox_label.grid(row=4, column=0, sticky="w", pady=10, padx=10)

listbox = tk.Listbox(window, selectmode=tk.MULTIPLE, width=20, height=5)
listbox.grid(row=5, column=0, sticky="w", padx=10)

items = ["K-means", "GMM", "K-means Silhouette", "GMM Silhouette"]

# Add items to the listbox
for item in items:
    listbox.insert(tk.END, item)

# Create the "Run" button
run_button = tk.Button(window, text="Run", command=run_function, height=2, width=10, bg="#6c7b8b", fg="white")
run_button.grid(row=6, column=0, sticky="w", pady=10, padx=10)

# Create the correlation label
correlation_label = tk.Label(window, text="Correlation coefficient:")
correlation_label.grid(row=3, column=0, sticky="w", pady=10)

# Set the correlation label to initially hidden
correlation_label.grid_remove()

# Start the Tkinter event loop
window.mainloop()
