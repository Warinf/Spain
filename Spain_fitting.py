import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.optimize import curve_fit

def time_to_hours(time_str):
    match = re.match(r'(\d+)\s*h\s*(\d+)?\s*min?', str(time_str))
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours + minutes / 60.0
    return np.nan

def custom_sigmoid(x, A1, A2, x0, dx):
    return (A1 - A2) / (1 + np.exp((x - x0) / dx)) + A2

def process_column(df, column_name, time_range=None):
    xdata = df['Time']
    ydata = df[column_name]
    
    if time_range:
        mask = (xdata >= time_range[0]) & (xdata <= time_range[1])
        xdata = xdata[mask]
        ydata = ydata[mask]
    
    max_value, min_value = np.max(ydata), np.min(ydata)
    initial_guess = [max_value, min_value, np.mean(xdata), 1.0]
    
    try:
        popt, _ = curve_fit(custom_sigmoid, xdata, ydata, p0=initial_guess, method='dogbox', maxfev=10000)
        return popt, xdata, ydata
    except:
        return None, xdata, ydata

st.title("Fluorescence Data Analysis")
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file, skiprows=2)
    df_raw.columns = ['Content', 'Time'] + [f"Sample X{i}" for i in range(1, len(df_raw.columns) - 1)]
    df_raw['Time'] = df_raw['Time'].apply(lambda x: time_to_hours(str(x)))
    df_raw.dropna(subset=['Time'], inplace=True)
    fluorescence_cols = [col for col in df_raw.columns if col.startswith('Sample')]
    df_raw[fluorescence_cols] = (df_raw[fluorescence_cols] - df_raw[fluorescence_cols].min()) / (
        df_raw[fluorescence_cols].max() - df_raw[fluorescence_cols].min()
    )
    
    time_range = st.slider("Select Time Range for Fitting (hours)", 0.0, float(df_raw['Time'].max()), (0.0, float(df_raw['Time'].max())))
    results = []
    
    for col in fluorescence_cols:
        popt, xdata, ydata = process_column(df_raw, col, time_range)
      if popt is not None:
       A1, A2, x0, dx = popt
        t50, tlag = x0, x0 - (2 * dx)
    if t50 >= 0 and tlag >= 0:
        results.append({"Sample": col, "Half-Time (hours)": t50, "Lag-Time (hours)": tlag})
else:
    st.warning(f"Curve fitting failed for {col}. Skipping this sample.")

    
    if results:
        df_results = pd.DataFrame(results)
        st.write("### Results Table")
        st.dataframe(df_results)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(results))
        width = 0.35
        
        ax.bar(x - width / 2, df_results["Half-Time (hours)"], width, label='Half-Time', color='b')
        ax.bar(x + width / 2, df_results["Lag-Time (hours)"], width, label='Lag-Time', color='r')
        ax.set_xlabel("Sample")
        ax.set_ylabel("Time (hours)")
        ax.set_xticks(x)
        ax.set_xticklabels(df_results["Sample"], rotation=90)
        ax.legend()
        
        st.pyplot(fig)
        
        output_excel = df_results.to_excel("Results.xlsx", index=False)
        st.download_button("Download Results", data=output_excel, file_name="Results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
