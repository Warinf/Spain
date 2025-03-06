import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.optimize import curve_fit
import sys

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the input and output file paths relative to the script directory
input_file = os.path.join(os.getcwd(), 'Data.xlsx')  # Excel file in the same folder as the script
if hasattr(sys, '_MEIPASS'):  # Running from PyInstaller executable
    base_dir = sys._MEIPASS
else:  # Running as a script
    base_dir = os.path.dirname(os.path.abspath(__file__))

output_file = os.path.join(os.path.dirname(sys.executable), 'Results.xlsx') if hasattr(sys, '_MEIPASS') else os.path.join(base_dir, 'Results.xlsx')

# Extract the directory from the input file path
input_dir = os.path.dirname(input_file)

# Load and preprocess the data, starting from row 2 (3rd row in 0-based indexing)
df_raw = pd.read_excel(input_file, skiprows=2)  # Skip the first two rows (metadata)

# Set proper headers: The first row after skipping is now the header row
df_raw.columns = ['Content', 'Time'] + [f"Sample X{i}" for i in range(1, len(df_raw.columns) - 1)]

# Function to convert time to hours
def time_to_hours(time_str):
    # Regular expression to match hours and minutes (e.g., "0 h", "0 h 10 min")
    match = re.match(r'(\d+)\s*h\s*(\d+)?\s*min?', time_str)
    if match:
        hours = int(match.group(1))  # Extract hours
        minutes = int(match.group(2)) if match.group(2) else 0  # Extract minutes (default to 0 if not present)
        return hours + minutes / 60.0  # Convert to hours
    else:
        return np.nan  # Return NaN if the time format is incorrect

# Apply the time_to_hours function to convert all 'Time' values
df_raw['Time'] = df_raw['Time'].apply(lambda x: time_to_hours(str(x)))

# Convert 'Time' column to numeric, forcing errors to NaN and dropping them
df_raw['Time'] = pd.to_numeric(df_raw['Time'], errors='coerce')

# Drop rows where 'Time' is NaN
df_raw = df_raw.dropna(subset=['Time'])

# Normalize fluorescence data
fluorescence_cols = [col for col in df_raw.columns if col.startswith('Sample')]
df_raw[fluorescence_cols] = (df_raw[fluorescence_cols] - df_raw[fluorescence_cols].min()) / (
    df_raw[fluorescence_cols].max() - df_raw[fluorescence_cols].min()
)

# Define the custom sigmoid function
def custom_sigmoid(x, A1, A2, x0, dx):
    return (A1 - A2) / (1 + np.exp((x - x0) / dx)) + A2

# Lists to store Half-Time and Lag-Time results
results = []

# Function to process each column
def process_column(column_name, ax, idx, time_range=None):
    xdata = df_raw['Time']
    ydata = df_raw[column_name]

    # Mask the data based on the time range (if provided)
    if time_range:
        mask = (xdata >= time_range[0]) & (xdata <= time_range[1])
        xdata = xdata[mask]
        ydata = ydata[mask]

    # Initial guess for curve fitting
    max_value = np.max(ydata)
    min_value = np.min(ydata)
    initial_guess = [max_value, min_value, np.mean(xdata), 1.0]

    try:
        # Perform curve fitting
        popt, _ = curve_fit(custom_sigmoid, xdata, ydata, p0=initial_guess, method='dogbox', maxfev=10000)

        # Plot data and fit in the corresponding subplot
        x_fit = np.linspace(0, np.max(xdata), len(xdata))
        y_fit = custom_sigmoid(x_fit, *popt)
        
        # Plot the raw data and fit, without showing the legend
        ax.plot(xdata, ydata, 'bo', label='Data')  # Blue markers for data points
        ax.plot(x_fit, y_fit, 'r-')  # Red line for fit
        ax.set_title(column_name)

        # Extract parameters and calculate metrics
        A1, A2, x0, dx = popt
        t50 = x0
        tlag = t50 - (2 * dx)

        # Skip if half-time or lag-time are negative
        if t50 < 0 or tlag < 0:
            print(f"Skipping {column_name} due to negative Half-Time or Lag-Time.")
            return  # Skip this sample if the times are negative

        # Append results
        results.append({
            "Sample": column_name,
            "Half-Time (hours)": t50,
            "Lag-Time (hours)": tlag
        })

    except Exception as e:
        print(f"Error processing {column_name}: {e}")

# Ask the user for the time range to mask the data (in hours)
print("Please enter the time range for fitting (in hours).")
start_time = float(input("Enter start time (e.g., 0): "))
end_time = float(input("Enter end time (e.g., 10): "))
time_range = (start_time, end_time)

# Calculate the number of rows and columns for subplots
num_samples = len(fluorescence_cols)
max_plots = 96
cols = 8  # We will use 8 columns per row, and dynamically calculate rows based on the number of samples
rows = (num_samples // cols) + (1 if num_samples % cols > 0 else 0)

# Create subplots for the sigmoid fits
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes = axes.flatten()  # Flatten the 2D array of axes into a 1D array

# Process each fluorescence data column and plot in a subplot
for idx, col in enumerate(fluorescence_cols):
    process_column(col, axes[idx], idx, time_range)

# Adjust layout to prevent overlapping
plt.tight_layout()

# Save the subplot figure in the same directory as the input file
subplot_fig_path = os.path.join(input_dir, 'sigmoid_fits.png')
plt.savefig(subplot_fig_path)
print(f"Subplots saved to {subplot_fig_path}")

# Plot the results in a bar chart
fig_results, ax_results = plt.subplots(figsize=(10, 6))
width = 0.35  # Width of the bars
x = np.arange(len(results))  # Sample names

# Plot Half-Time and Lag-Time as bars
ax_results.bar(x - width / 2, [r["Half-Time (hours)"] for r in results], width, label='Half-Time', color='b')
ax_results.bar(x + width / 2, [r["Lag-Time (hours)"] for r in results], width, label='Lag-Time', color='r')

# Add labels, title, and legend
ax_results.set_xlabel('Sample')
ax_results.set_ylabel('Time (hours)')
ax_results.set_title('Half-Time and Lag-Time for Each Sample')
ax_results.set_xticks(x)
ax_results.set_xticklabels([r["Sample"] for r in results], rotation=90)
ax_results.legend()

# Save the bar plot figure in the same directory as the input file
bar_plot_fig_path = os.path.join(input_dir, 'half_lag_times.png')
plt.savefig(bar_plot_fig_path)
print(f"Bar plot saved to {bar_plot_fig_path}")

# Show the bar plot
plt.tight_layout()
plt.show()

# Convert results to a DataFrame and save to Excel
df_results = pd.DataFrame(results)
df_results.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")
