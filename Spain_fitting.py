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

def find_t50(xdata, ydata):
    try:
        idx = np.where(ydata >= 0.5)[0][0]  # Find first occurrence where intensity >= 0.5
        return xdata.iloc[idx] if idx < len(xdata) else None
    except:
        return None

def process_column(df, column_name, t_fit):
    xdata = df['Time']
    ydata = df[column_name]
    
    if t_fit is not None:
        mask = (xdata >= 0) & (xdata <= 2 * t_fit)  # Dynamic fitting window
        xdata = xdata[mask]
        ydata = ydata[mask]
    
    max_value, min_value = np.max(ydata), np.min(ydata)
    initial_guess = [max_value, min_value, np.mean(xdata), 1.0]
    
    try:
        popt, _ = curve_fit(custom_sigmoid, xdata, ydata, p0=initial_guess, method='dogbox', maxfev=10000)
        return popt, xdata, ydata
    except Exception as e:
        st.error(f"Error fitting data for {column_name}: {e}")
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
    
    results = []
    
    for col in fluorescence_cols:
        t50 = find_t50(df_raw['Time'], df_raw[col])
        if t50 is not None:
            popt, xdata, ydata = process_column(df_raw, col, t50)
            if popt is not None:
                A1, A2, x0, dx = popt
                tlag = x0 - (2 * dx)  # Compute lag time
                results.append({"Sample": col, "Half-Time (hours)": t50, "Lag-Time (hours)": tlag})
    
    if results:
        df_results = pd.DataFrame(results)
        st.write("### Results Table")
        st.dataframe(df_results)
        
        num_samples = len(fluorescence_cols)
        cols = 8
        rows = (num_samples // cols) + (1 if num_samples % cols > 0 else 0)
        fig, axes = plt.subplots(rows, cols, dpi=300)  # Increased size and resolution
        axes = axes.flatten()
        
        for idx, col in enumerate(fluorescence_cols):
            t50 = find_t50(df_raw['Time'], df_raw[col])
            if t50 is not None:
                popt, xdata, ydata = process_column(df_raw, col, t50)
                if popt is not None:
                    A1, A2, x0, dx = popt
                    ax = axes[idx]
                    ax.plot(xdata, ydata, 'b-', label=f"Data: {col}")
                    ax.plot(xdata, custom_sigmoid(xdata, *popt), 'r--', label="Sigmoid Fit")
                    ax.set_title(f"Sigmoid Fit - {col}")
                    ax.set_xlabel("Time (hours)")
                    ax.set_ylabel("Normalized Fluorescence")
                    ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        fig, ax = plt.subplots(figsize=(12, 8), dpi=200)  # Increased size and resolution
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
