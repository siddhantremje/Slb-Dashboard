import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
from pathlib import Path
from sklearn.cluster import KMeans
import time
from fpdf import FPDF

# --- Configuration Section ---
DATA_DIRECTORY = "." 
NUM_CLUSTERS = 3 
LOGO_FILE_NAME = "ntu_logo.png"

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Battery Health Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Function for Battery Icon ---
def get_battery_icon(health_category):
    """Returns an emoji battery icon based on health category."""
    if health_category == "Good":
        return "🟢🔋"
    elif health_category == "Marginal":
        return "🟡🔋"
    elif health_category == "Bad":
        return "🔴🔋"
    else:
        return "⚪🔋"

# --- Dashboard Title and Optional Logo ---
title_col, logo_col = st.columns([10, 2])

with title_col:
    st.title("🔋 Second-Life EV Battery Screening Dashboard")

with logo_col:
    if Path(LOGO_FILE_NAME).exists():
        st.image(LOGO_FILE_NAME, width=90)
    else:
        st.info(f"Logo file '{LOGO_FILE_NAME}' not found. Place it in the same directory for display.")

st.markdown("""
This dashboard is designed to **screen and classify second-life EV batteries** based on their energy potential and health,
helping to identify which batteries are most suitable for reuse in stationary energy storage systems.
""")

# --- Data Loading and Preprocessing Function ---
@st.cache_data
def load_and_process_battery_data(folder_path):
    """
    Loads battery data from multiple CSV files in the specified folder.
    Each file is expected to contain raw time-series data for one battery.
    Calculates average voltage, temperature, and total capacity.
    """
    all_battery_records = []
    
    start_time = time.time() 

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".csv"):
            file_full_path = os.path.join(folder_path, file_name)
            try:
                df = pd.read_csv(file_full_path)

                required_columns = ["Voltage_measured", "Temperature_measured", "Current_measured", "Time"]
                
                if not all(col in df.columns for col in required_columns):
                    missing = [col for col in required_columns if col not in df.columns]
                    st.warning(f"Skipped '{file_name}': Missing one or more required columns ({', '.join(missing)}). "
                               f"Please ensure CSVs have: {', '.join(required_columns)}.")
                    continue
                
                for col in required_columns:
                    if pd.api.types.is_object_dtype(df[col]):
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except ValueError:
                            st.warning(f"Non-numeric data found in '{col}' for file '{file_name}'. Coercing to numeric, NaNs may result.")
                df.dropna(subset=required_columns, inplace=True)

                if df.empty:
                    st.warning(f"Skipped '{file_name}': No valid numeric data after cleaning.")
                    continue

                avg_voltage = df["Voltage_measured"].mean()
                avg_temperature = df["Temperature_measured"].mean()
                
                avg_soc = df.get("SOC_Percent", np.nan).mean() if "SOC_Percent" in df.columns else np.nan
                avg_degradation_rate = df.get("Degradation_Rate_Percent", np.nan).mean() if "Degradation_Rate_Percent" in df.columns else np.nan
                
                current_values = df["Current_measured"].values
                time_values = df["Time"].values
                
                if time_values.max() < 1000 and time_values.max() > 0: 
                    time_values = time_values * 60 
                
                if len(time_values) < 2 or len(current_values) < 2:
                    st.warning(f"Skipped '{file_name}': Not enough time points for capacity calculation.")
                    total_capacity_ah = np.nan
                else:
                    delta_t = np.diff(time_values)
                    avg_current_intervals = (current_values[:-1] + current_values[1:]) / 2
                    total_capacity_ah = np.sum(np.abs(avg_current_intervals * delta_t)) / 3600
                
                all_battery_records.append([
                    file_name, 
                    avg_voltage, 
                    avg_temperature, 
                    total_capacity_ah, 
                    avg_soc, 
                    avg_degradation_rate
                ])
            except Exception as e:
                st.error(f"Error processing '{file_name}': {e}. Skipping this file.")
                
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)
    
    st.session_state['processing_time'] = processing_time
    st.session_state['last_refresh_time'] = time.strftime("%Y-%m-%d %H:%M:%S")

    return pd.DataFrame(all_battery_records, columns=[
        "File Name", 
        "Avg Voltage", 
        "Avg Temp", 
        "Capacity (Ah)", 
        "Avg SOC (%)", 
        "Avg Degradation Rate (%)"
    ]).dropna(subset=["Capacity (Ah)"])

# --- Function to load full data for a single battery file ---
@st.cache_data
def load_single_battery_data(file_name, folder_path):
    """Loads the full DataFrame for a single specified battery file."""
    file_full_path = os.path.join(folder_path, file_name)
    try:
        df = pd.read_csv(file_full_path)
        
        required_plot_cols = ["Voltage_measured", "Current_measured", "Temperature_measured", "Time"]
        if "SOC_Percent" in df.columns:
            required_plot_cols.append("SOC_Percent")

        for col in required_plot_cols:
            if col in df.columns and pd.api.types.is_object_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=[col for col in required_plot_cols if col in df.columns], inplace=True)

        if "Time" in df.columns and df["Time"].max() < 1000 and df["Time"].max() > 0:
             df["Time"] = df["Time"] * 60
        
        return df
    except Exception as e:
        st.error(f"Could not load detailed data for '{file_name}' from '{folder_path}': {e}")
        return pd.DataFrame()

# --- PDF Generation Function ---
def create_pdf_report(df_summary, selected_battery_summary=None, health_color_map=None, current_scatter_health_choice=None, current_pie_health_choice=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Battery Screening Report", 0, 1, "C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "C")
    pdf.ln(10)

    # --- Fleet Summary ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "1. Fleet Summary", 0, 1, "L")
    pdf.set_font("Arial", "", 10)
    
    pdf.multi_cell(0, 5, f"Total Batteries Processed: {len(df_summary)}")
    good_count = df_summary[df_summary["Health Category (User-Defined)"] == "Good"].shape[0]
    marginal_count = df_summary[df_summary["Health Category (User-Defined)"] == "Marginal"].shape[0]
    bad_count = df_summary[df_summary["Health Category (User-Defined)"] == "Bad"].shape[0]
    pdf.multi_cell(0, 5, f"Classified as Good (User-Defined): {good_count}")
    pdf.multi_cell(0, 5, f"Classified as Marginal (User-Defined): {marginal_count}")
    pdf.multi_cell(0, 5, f"Classified as Bad (User-Defined): {bad_count}")
    pdf.ln(5)

    # Add Average Metrics by Health Category
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 7, "Average Metrics by User-Defined Health Category:", 0, 1, "L")
    pdf.set_font("Arial", "", 8)
    
    agg_cols_pdf = {
        'Avg_Capacity':('Capacity (Ah)', 'mean'),
        'Avg_Voltage':('Avg Voltage', 'mean'),
        'Avg_Temp':('Avg Temp', 'mean')
    }
    if 'Avg SOC (%)' in df_summary.columns:
        agg_cols_pdf['Avg_SOC'] = ('Avg SOC (%)', 'mean')
    if 'Avg Degradation Rate (%)' in df_summary.columns:
        agg_cols_pdf['Avg_Degradation'] = ('Avg Degradation Rate (%)', 'mean')

    avg_metrics_by_category = df_summary.groupby("Health Category (User-Defined)").agg(**agg_cols_pdf).round(2).reset_index()

    headings = list(avg_metrics_by_category.columns)
    data = avg_metrics_by_category.values.tolist()

    pdf.set_fill_color(200, 220, 255)
    pdf.set_draw_color(128, 128, 128)
    pdf.set_line_width(0.3)

    col_widths = [30, 25, 25, 25] 
    if 'Avg_SOC' in headings:
        col_widths.append(25)
    if 'Avg_Degradation' in headings:
        col_widths.append(35)
    
    total_width = sum(col_widths)
    if total_width > 190: 
        scale_factor = 190 / total_width
        col_widths = [w * scale_factor for w in col_widths]


    for col, width in zip(headings, col_widths):
        pdf.cell(width, 7, col, 1, 0, 'C', 1)
    pdf.ln()

    pdf.set_fill_color(240, 240, 240) 
    fill = False
    for row in data:
        for item, width in zip(row, col_widths):
            pdf.cell(width, 6, str(item), 1, 0, 'C', fill)
        pdf.ln()
        fill = not fill
    pdf.ln(10)

    # --- Note about plots not being embedded ---
    pdf.set_font("Arial", "I", 8)
    pdf.multi_cell(0, 5, "Note: Interactive plots (Capacity Trend, Scatter Plot, Pie Chart) are available in the dashboard itself. For simplicity and to avoid external dependencies, they are not embedded in this PDF report.")
    pdf.ln(5)

    # --- Detailed Battery Report (if selected) ---
    if selected_battery_summary is not None:
        pdf.add_page() # New page for detailed report
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"2. Detailed Report for Battery: {selected_battery_summary['File Name']}", 0, 1, "L")
        pdf.set_font("Arial", "", 10)
        pdf.ln(5)

        # KPIs
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 7, "2.1 Key Performance Indicators (KPIs):", 0, 1, "L")
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 5, f"Average Voltage: {selected_battery_summary['Avg Voltage']:.2f} V")
        pdf.multi_cell(0, 5, f"Average Temperature: {selected_battery_summary['Avg Temp']:.2f} °C")
        pdf.multi_cell(0, 5, f"Capacity: {selected_battery_summary['Capacity (Ah)']:.2f} Ah")
        if 'Avg SOC (%)' in selected_battery_summary:
            pdf.multi_cell(0, 5, f"Average SOC: {selected_battery_summary['Avg SOC (%)']:.2f} %")
        if 'Avg Degradation Rate (%)' in selected_battery_summary:
            pdf.multi_cell(0, 5, f"Average Degradation Rate: {selected_battery_summary['Avg Degradation Rate (%)']:.2f} %")
        pdf.multi_cell(0, 5, f"Anomaly Status: {'Anomaly Detected!' if selected_battery_summary['Is Anomaly'] else 'No Anomaly'}")
        pdf.ln(5)

        # Classification Criteria
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 7, "2.2 Classification Criteria & Result:", 0, 1, "L")
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 5, f"KMeans Health Status: {selected_battery_summary['Health Category (KMeans)']}")
        pdf.multi_cell(0, 5, f"User-Defined Health Status: {selected_battery_summary['Health Category (User-Defined)']}")
        pdf.multi_cell(0, 5, "User-Defined Thresholds Used:")
        pdf.multi_cell(0, 5, f"  - Capacity: >= {st.session_state.get('cap_good_slider', 'N/A')} Ah (Good), >= {st.session_state.get('cap_marginal_slider', 'N/A')} Ah (Marginal)")
        pdf.multi_cell(0, 5, f"  - Voltage: >= {st.session_state.get('volt_good_slider', 'N/A')} V (Good), >= {st.session_state.get('volt_marginal_slider', 'N/A')} V (Marginal)")
        pdf.multi_cell(0, 5, f"  - Temperature: Between {st.session_state.get('temp_low_bad_slider', 'N/A')} and {st.session_state.get('temp_high_bad_slider', 'N/A')} °C (Good)")
        if 'Avg Degradation Rate (%)' in selected_battery_summary:
            pdf.multi_cell(0, 5, f"  - Degradation Rate: <= {st.session_state.get('degrad_good_slider', 'N/A')}% (Good), <= {st.session_state.get('degrad_marginal_slider', 'N/A')}% (Marginal)")
        pdf.ln(5)

        # Recommendation
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 7, "2.3 Recommendation:", 0, 1, "L")
        pdf.set_font("Arial", "", 10)
        user_defined_status = selected_battery_summary['Health Category (User-Defined)']
        if user_defined_status == "Good":
            pdf.multi_cell(0, 5, "This battery shows high potential for efficient energy provision and is suitable for repurposing in stationary energy storage systems.")
        elif user_defined_status == "Marginal":
            pdf.multi_cell(0, 5, "This battery has marginal potential. It may require further detailed testing or could be suitable for less demanding applications.")
        else: # Bad
            pdf.multi_cell(0, 5, "This battery is classified as low potential. It is likely unsuitable for repurposing and should be considered for recycling.")
        pdf.ln(5)

        # Conceptual Techno-Economic Implications
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 7, "2.4 Conceptual Techno-Economic Implications:", 0, 1, "L")
        pdf.set_font("Arial", "", 10)
        if user_defined_status == "Good":
            pdf.multi_cell(0, 5, "Based on its 'Good' health, this battery likely has a **high economic value** for repurposing, potentially leading to a low Levelized Cost of Storage (LCOS) and positive Net Present Value (NPV) in a grid-scale system.")
            pdf.multi_cell(0, 5, f"  - Simulated Expected Second-Life Duration: {st.session_state.get(f'duration_good_{selected_battery_summary['File Name']}', 'N/A')} Years")
            pdf.multi_cell(0, 5, f"  - Simulated Assumed Energy Price: £{st.session_state.get(f'price_good_{selected_battery_summary['File Name']}', 'N/A'):.2f}/kWh")
            # Recalculate for PDF to match dashboard logic
            expected_duration_val = st.session_state.get(f'duration_good_{selected_battery_summary["File Name"]}', 7)
            simulated_lcos_val = 0.10 - (selected_battery_summary['Capacity (Ah)'] * 0.001) + (expected_duration_val * 0.001)
            simulated_npv_val = 1000 + (selected_battery_summary['Capacity (Ah)'] * 50) - (expected_duration_val * 10)
            pdf.multi_cell(0, 5, f"  - Simulated LCOS: £{simulated_lcos_val:.2f}/kWh")
            pdf.multi_cell(0, 5, f"  - Simulated NPV: £{simulated_npv_val:.2f}")

        elif user_defined_status == "Marginal":
            pdf.multi_cell(0, 5, "With 'Marginal' health, the economic viability of this battery for repurposing might be **moderate**. Further testing is recommended to assess its true LCOS and NPV, or it might be suited for less critical applications.")
            pdf.multi_cell(0, 5, f"  - Simulated Expected Second-Life Duration: {st.session_state.get(f'duration_marginal_{selected_battery_summary['File Name']}', 'N/A')} Years")
            pdf.multi_cell(0, 5, f"  - Simulated Assumed Energy Price: £{st.session_state.get(f'price_marginal_{selected_battery_summary['File Name']}', 'N/A'):.2f}/kWh")
            expected_duration_val = st.session_state.get(f'duration_marginal_{selected_battery_summary["File Name"]}', 3)
            simulated_lcos_val = 0.25 - (selected_battery_summary['Capacity (Ah)'] * 0.001) + (expected_duration_val * 0.005)
            simulated_npv_val = 500 + (selected_battery_summary['Capacity (Ah)'] * 20) - (expected_duration_val * 20)
            pdf.multi_cell(0, 5, f"  - Simulated LCOS: £{simulated_lcos_val:.2f}/kWh")
            pdf.multi_cell(0, 5, f"  - Simulated NPV: £{simulated_npv_val:.2f}")

        else: # Bad
            pdf.multi_cell(0, 5, "A 'Bad' health classification suggests **low economic viability** for repurposing. Its LCOS would likely be high, and NPV negative. It is primarily a candidate for material recovery through recycling.")
            pdf.multi_cell(0, 5, "  - Repurposing this battery is likely not economically viable. Focus on recycling value.")
        pdf.ln(5)

        # Sustainability/Disposal for Bad Batteries
        if user_defined_status == "Bad":
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 7, "2.5 Sustainability & Disposal Recommendations (UK Context):", 0, 1, "L")
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 5, """
            For batteries classified as 'Bad' and unsuitable for second-life applications, responsible recycling is crucial for sustainability in the UK. Key considerations include:
            -   **Specialized Recycling Facilities:** Engage with approved battery recycling facilities in the UK (e.g., those part of the BatteryBack scheme, or accredited waste management companies).
            -   **Material Recovery:** Focus on facilities that maximize the recovery of valuable materials (lithium, cobalt, nickel, manganese) to reduce reliance on virgin resources.
            -   **Environmental Compliance:** Ensure disposal adheres to UK and EU waste electrical and electronic equipment (WEEE) regulations and hazardous waste directives.
            -   **Reduced Landfill Impact:** Proper recycling prevents hazardous materials from contaminating landfills and reduces the overall environmental footprint of EV batteries.
            """)
            pdf.ln(5)

    return pdf.output(dest='S').encode('latin-1') # Return as bytes

# --- Main Application Logic ---

# Initialize session state for tour if not present
if 'tour_step' not in st.session_state:
    st.session_state.tour_step = 0

# Initialize selected_file in session state for programmatic control
if 'pre_selected_sidebar_battery' not in st.session_state:
    st.session_state.pre_selected_sidebar_battery = "All Batteries (Summary)"


data_folder_exists = os.path.exists(DATA_DIRECTORY)
csv_files_present = False
if data_folder_exists:
    for f in os.listdir(DATA_DIRECTORY):
        if os.path.isfile(os.path.join(DATA_DIRECTORY, f)) and f.lower().endswith(".csv"):
            csv_files_present = True
            break

if csv_files_present: # Only proceed if CSVs are found
    # --- Animation: Spinner while loading data ---
    with st.spinner("Loading and processing battery data..."):
        df_dashboard_data = load_and_process_battery_data(DATA_DIRECTORY)
    
    if df_dashboard_data.empty:
        st.warning("No valid battery data could be loaded from the CSV files. Please check file contents and column names.")
    else:
        st.success(f"Battery data loaded and processed successfully! (Took {st.session_state.get('processing_time', 'N/A')} seconds)")

        # --- Dashboard Tour Logic ---
        if st.session_state.tour_step == 0:
            if st.button("Start Dashboard Tour 🚀", key="start_tour_button"):
                st.session_state.tour_step = 1
                st.rerun() 

        if st.session_state.tour_step > 0:
            tour_messages = {
                1: "Welcome to the **Second-Life EV Battery Screening Dashboard**! This tool helps classify batteries for repurposing.",
                2: "On the **left sidebar**, you'll find 'Health Classification Settings'. Here, you can adjust thresholds for 'Good', 'Marginal', and 'Bad' batteries based on Capacity, Voltage, Temperature, and Degradation Rate. Try moving the sliders!",
                3: "Below that, the 'Anomaly Detection' section flags any batteries that are statistical outliers in terms of capacity.",
                4: "Further down, 'Plot Color Customization' allows you to personalize the chart colors for each health category.",
                5: "Finally, the 'Detailed Battery View' lets you select individual batteries or compare multiple ones.",
                6: "In the **main content area**, the 'Fleet Health Summary' provides an immediate overview of your battery fleet based on your chosen thresholds.",
                7: "The 'Average Metrics by Health Category' table gives you quantitative insights into each classified group.",
                8: "The 'Classified Batteries Overview' table allows you to filter and inspect batteries by their health status from either KMeans or your User-Defined settings.",
                9: "The 'Fleet-wide Visualizations' (Capacity Trend, Scatter Plot, Pie Chart) offer interactive ways to explore overall fleet health.",
                10: "When you select an individual battery from the sidebar, you'll see its specific KPIs, a clear recommendation, and conceptual links to Techno-Economic implications.",
                11: "You can also compare multiple batteries' time-series data side-by-side in the 'Multi-Battery Comparison' view.",
                12: "That concludes the tour! Feel free to explore. You can download the summary data at any time.",
            }
            
            if st.session_state.tour_step <= len(tour_messages):
                st.info(tour_messages[st.session_state.tour_step])
                if st.button("Next Step »", key="next_tour_step"):
                    st.session_state.tour_step += 1
                    st.rerun() 
            else:
                st.session_state.tour_step = 0 # Reset tour
                st.rerun() 


        # --- Battery Health Clustering (KMeans) ---
        if len(df_dashboard_data) < NUM_CLUSTERS:
            st.info(f"Not enough unique battery files ({len(df_dashboard_data)}) for {NUM_CLUSTERS} clusters. Skipping KMeans categorization.")
            df_dashboard_data["Cluster"] = "N/A"
            df_dashboard_data["Health Category (KMeans)"] = "Not enough data"
        else:
            features_for_clustering = df_dashboard_data[["Avg Voltage", "Avg Temp", "Capacity (Ah)"]]
            
            if features_for_clustering.drop_duplicates().shape[0] < NUM_CLUSTERS or (features_for_clustering.std() == 0).any(): 
                st.warning(f"Insufficient variation or unique data points ({features_for_clustering.drop_duplicates().shape[0]} unique) for meaningful KMeans clustering with {NUM_CLUSTERS} clusters. All batteries might be too similar. Skipping KMeans categorization.")
                df_dashboard_data["Cluster"] = "N/A"
                df_dashboard_data["Health Category (KMeans)"] = "Not enough data variation"
            else:
                try:
                    kmeans_model = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init='auto')
                    df_dashboard_data["Cluster"] = kmeans_model.fit_predict(features_for_clustering)
                    
                    cluster_centers = kmeans_model.cluster_centers_
                    capacity_order = np.argsort(cluster_centers[:, 2])[::-1]
                    
                    health_labels_map = {}
                    if NUM_CLUSTERS == 3:
                        if len(capacity_order) >= 3:
                            health_labels_map[capacity_order[0]] = "Good"
                            health_labels_map[capacity_order[1]] = "Marginal"
                            health_labels_map[capacity_order[2]] = "Bad"
                        elif len(capacity_order) == 2:
                            health_labels_map[capacity_order[0]] = "Good"
                            health_labels_map[capacity_order[1]] = "Bad"
                        elif len(capacity_order) == 1:
                            health_labels_map[capacity_order[0]] = "Good"
                        else:
                            for i, cluster_id in enumerate(capacity_order):
                                health_labels_map[cluster_id] = f"Cluster {i+1}"
                    else: 
                        if len(capacity_order) >= 2:
                            health_labels_map[capacity_order[0]] = "Good"
                            health_labels_map[capacity_order[1]] = "Bad"
                        elif len(capacity_order) == 1:
                            health_labels_map[capacity_order[0]] = "Good"
                        else:
                            for i, cluster_id in enumerate(capacity_order):
                                health_labels_map[cluster_id] = f"Cluster {i+1}"
                    
                    df_dashboard_data["Health Category (KMeans)"] = df_dashboard_data["Cluster"].map(health_labels_map).fillna("Unknown")
                
                except Exception as e:
                    st.error(f"Error during KMeans clustering: {e}. This might be due to problematic data or an unexpected edge case in clustering. Please check your data's distribution.")
                    df_dashboard_data["Cluster"] = "Error"
                    df_dashboard_data["Health Category (KMeans)"] = "Clustering Failed"
        
        # --- Sidebar for Health Classification Settings ---
        st.sidebar.header("Health Classification Settings")
        st.sidebar.markdown("Define your own criteria for battery health:")
        
        # --- Multi-Metric User-Defined Thresholds ---
        st.sidebar.subheader("User-Defined Thresholds")
        st.sidebar.markdown("Set thresholds for Capacity, Voltage, Temperature, and Degradation Rate to classify batteries.")

        min_cap = df_dashboard_data['Capacity (Ah)'].min() if not df_dashboard_data.empty else 0
        max_cap = df_dashboard_data['Capacity (Ah)'].max() if not df_dashboard_data.empty else 10
        min_volt = df_dashboard_data['Avg Voltage'].min() if not df_dashboard_data.empty else 0
        max_volt = df_dashboard_data['Avg Voltage'].max() if not df_dashboard_data.empty else 5
        min_temp = df_dashboard_data['Avg Temp'].min() if not df_dashboard_data.empty else 0
        max_temp = df_dashboard_data['Avg Temp'].max() if not df_dashboard_data.empty else 50
        
        if 'Avg Degradation Rate (%)' in df_dashboard_data.columns and not df_dashboard_data['Avg Degradation Rate (%)'].isnull().all():
            min_degrad = df_dashboard_data['Avg Degradation Rate (%)'].min()
            max_degrad = df_dashboard_data['Avg Degradation Rate (%)'].max()
        else:
            min_degrad = 0.0
            max_degrad = 100.0 

        # Capacity Sliders
        cap_good_threshold = st.sidebar.slider(
            "Good (above)", min_value=float(min_cap), max_value=float(max_cap), 
            value=float(st.session_state.get("cap_good_slider", max_cap * 0.9 if max_cap > 0 else 9.0)), step=0.01, format="%.2f", key="cap_good_slider"
        )
        cap_marginal_threshold = st.sidebar.slider(
            "Marginal (above, below Good)", min_value=float(min_cap), max_value=float(cap_good_threshold), 
            value=float(st.session_state.get("cap_marginal_slider", max_cap * 0.7 if max_cap > 0 else 7.0)), step=0.01, format="%.2f", key="cap_marginal_slider"
        )

        # Voltage Sliders (Higher voltage is generally better)
        st.sidebar.markdown("**Average Voltage (V) Thresholds:**")
        volt_good_threshold = st.sidebar.slider(
            "Good (above)", min_value=float(min_volt), max_value=float(max_volt), 
            value=float(st.session_state.get("volt_good_slider", max_volt * 0.9 if max_volt > 0 else 4.0)), step=0.01, format="%.2f", key="volt_good_slider"
        )
        volt_marginal_threshold = st.sidebar.slider(
            "Marginal (above, below Good)", min_value=float(min_volt), max_value=float(volt_good_threshold), 
            value=float(st.session_state.get("volt_marginal_slider", max_volt * 0.7 if max_volt > 0 else 3.5)), step=0.01, format="%.2f", key="volt_marginal_slider"
        )

        # Temperature Sliders (Optimal range, so we define a "bad" range)
        st.sidebar.markdown("**Average Temperature (°C) Thresholds (Optimal Range):**")
        temp_low_bad_threshold = st.sidebar.slider(
            "Too Cold (below)", min_value=float(min_temp), max_value=float(max_temp), 
            value=float(st.session_state.get("temp_low_bad_slider", max_temp * 0.2 if max_temp > 0 else 5.0)), step=0.01, format="%.2f", key="temp_low_bad_slider"
        )
        temp_high_bad_threshold = st.sidebar.slider(
            "Too Hot (above)", min_value=float(min_temp), max_value=float(max_temp), 
            value=float(st.session_state.get("temp_high_bad_slider", max_temp * 0.8 if max_temp > 0 else 35.0)), step=0.01, format="%.2f", key="temp_high_bad_slider"
        )

        # Degradation Rate Sliders (Lower degradation is better) - Only show if column exists
        if 'Avg Degradation Rate (%)' in df_dashboard_data.columns:
            st.sidebar.markdown("**Degradation Rate (%) Thresholds:**")
            degrad_good_threshold = st.sidebar.slider(
                "Good (below)", min_value=float(min_degrad), max_value=float(max_degrad), 
                value=float(st.session_state.get("degrad_good_slider", max_degrad * 0.1 if max_degrad > 0 else 5.0)), step=0.01, format="%.2f", key="degrad_good_slider"
            )
            degrad_marginal_threshold = st.sidebar.slider(
                "Marginal (below, above Good)", min_value=float(degrad_good_threshold), max_value=float(max_degrad), 
                value=float(st.session_state.get("degrad_marginal_slider", max_degrad * 0.3 if max_degrad > 0 else 15.0)), step=0.01, format="%.2f", key="degrad_marginal_slider"
            )
        else:
            degrad_good_threshold = np.inf 
            degrad_marginal_threshold = np.inf
            st.sidebar.info("Degradation Rate (%) data not found in CSVs for thresholding.")
        
        def apply_user_thresholds(row):
            is_cap_good = row['Capacity (Ah)'] >= cap_good_threshold
            is_cap_marginal = row['Capacity (Ah)'] >= cap_marginal_threshold and row['Capacity (Ah)'] < cap_good_threshold
            
            is_volt_good = row['Avg Voltage'] >= volt_good_threshold
            is_volt_marginal = row['Avg Voltage'] >= volt_marginal_threshold and row['Avg Voltage'] < volt_good_threshold
            
            is_temp_good = (row['Avg Temp'] > temp_low_bad_threshold) and (row['Avg Temp'] < temp_high_bad_threshold)
            
            is_degrad_good = True 
            is_degrad_marginal = False 
            if 'Avg Degradation Rate (%)' in row and not pd.isna(row['Avg Degradation Rate (%)']):
                is_degrad_good = row['Avg Degradation Rate (%)'] <= degrad_good_threshold
                is_degrad_marginal = row['Avg Degradation Rate (%)'] > degrad_good_threshold and row['Avg Degradation Rate (%)'] <= degrad_marginal_threshold

            if is_cap_good and is_volt_good and is_temp_good and is_degrad_good:
                return "Good"
            elif (is_cap_marginal or is_volt_marginal or is_degrad_marginal) and is_temp_good:
                return "Marginal"
            elif (row['Avg Temp'] <= temp_low_bad_threshold or row['Avg Temp'] >= temp_high_bad_threshold) or \
                 (row['Capacity (Ah)'] < cap_marginal_threshold) or \
                 (row['Avg Voltage'] < volt_marginal_threshold) or \
                 ( ('Avg Degradation Rate (%)' in row and not pd.isna(row['Avg Degradation Rate (%)'])) and row['Avg Degradation Rate (%)'] > degrad_marginal_threshold): 
                return "Bad"
            else:
                return "Unknown"
        
        df_dashboard_data["Health Category (User-Defined)"] = df_dashboard_data.apply(apply_user_thresholds, axis=1)

        # --- Reset User Thresholds Button ---
        if st.sidebar.button("Reset User Thresholds to Default", key="reset_thresholds_button"):
            st.session_state["cap_good_slider"] = float(max_cap * 0.9 if max_cap > 0 else 9.0)
            st.session_state["cap_marginal_slider"] = float(max_cap * 0.7 if max_cap > 0 else 7.0)
            st.session_state["volt_good_slider"] = float(max_volt * 0.9 if max_volt > 0 else 4.0)
            st.session_state["volt_marginal_slider"] = float(max_volt * 0.7 if max_volt > 0 else 3.5)
            st.session_state["temp_low_bad_slider"] = float(max_temp * 0.2 if max_temp > 0 else 5.0)
            st.session_state["temp_high_bad_slider"] = float(max_temp * 0.8 if max_temp > 0 else 35.0)
            if 'Avg Degradation Rate (%)' in df_dashboard_data.columns:
                st.session_state["degrad_good_slider"] = float(max_degrad * 0.1 if max_degrad > 0 else 5.0)
                st.session_state["degrad_marginal_slider"] = float(max_degrad * 0.3 if max_degrad > 0 else 15.0)
            st.rerun() 

        # --- Anomaly Detection (Outlier Flagging) ---
        st.sidebar.subheader("Anomaly Detection (Outliers)")
        st.sidebar.markdown("Flag batteries that are statistical outliers based on Capacity.")
        
        Q1 = df_dashboard_data['Capacity (Ah)'].quantile(0.25)
        Q3 = df_dashboard_data['Capacity (Ah)'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_dashboard_data['Is Anomaly'] = ((df_dashboard_data['Capacity (Ah)'] < lower_bound) | 
                                            (df_dashboard_data['Capacity (Ah)'] > upper_bound))
        
        if df_dashboard_data['Is Anomaly'].any():
            st.sidebar.warning("🚨 Anomalies detected in Capacity data!")
            anomalies = df_dashboard_data[df_dashboard_data['Is Anomaly']]
            st.sidebar.dataframe(anomalies[['File Name', 'Capacity (Ah)']], height=150)
        else:
            st.sidebar.info("No significant capacity anomalies detected.")

        # --- Sidebar for Plot Color Customization ---
        st.sidebar.subheader("Plot Color Customization")
        st.sidebar.markdown("Choose colors for health categories:")
        
        if 'good_color' not in st.session_state:
            st.session_state.good_color = "#28a745" # Green
        if 'marginal_color' not in st.session_state:
            st.session_state.marginal_color = "#ffc107" # Yellow/Orange
        if 'bad_color' not in st.session_state:
            st.session_state.bad_color = "#dc3545" # Red
        if 'unknown_color' not in st.session_state:
            st.session_state.unknown_color = "#6c757d" # Gray

        st.session_state.good_color = st.sidebar.color_picker("Good Category Color", st.session_state.good_color, key="good_color_picker")
        st.session_state.marginal_color = st.sidebar.color_picker("Marginal Category Color", st.session_state.marginal_color, key="marginal_color_picker")
        st.session_state.bad_color = st.sidebar.color_picker("Bad Category Color", st.session_state.bad_color, key="bad_color_picker")
        st.session_state.unknown_color = st.sidebar.color_picker("Unknown/N/A Color", st.session_state.unknown_color, key="unknown_color_picker")

        health_color_map = {
            "Good": st.session_state.good_color,
            "Marginal": st.session_state.marginal_color,
            "Bad": st.session_state.bad_color,
            "Unknown": st.session_state.unknown_color,
            "Not enough data": st.session_state.unknown_color,
            "Clustering Failed": st.session_state.unknown_color,
            "Not enough data variation": st.session_state.unknown_color,
            "N/A": st.session_state.unknown_color
        }

        # --- Sidebar for Battery Selection for Detail View ---
        st.sidebar.subheader("Detailed Battery View")
        
        # Determine the default index for the selectbox based on session state
        all_file_options = ["All Batteries (Summary)", "Multi-Battery Comparison"] + sorted(df_dashboard_data["File Name"].tolist())
        
        # Get the index of the pre-selected battery, default to 0 (All Batteries Summary)
        default_select_index = 0
        if 'pre_selected_sidebar_battery' in st.session_state and st.session_state.pre_selected_sidebar_battery in all_file_options:
            default_select_index = all_file_options.index(st.session_state.pre_selected_sidebar_battery)
            # Reset the pre-selected variable after using it, so it doesn't stick if user manually changes
            st.session_state.pre_selected_sidebar_battery = "All Batteries (Summary)" 

        selected_file = st.sidebar.selectbox(
            "Select a battery for detailed analysis:",
            options=all_file_options,
            index=default_select_index, # Set the index based on session state
            key="main_battery_select"
        )

        # --- Main Content Area ---
        if selected_file == "All Batteries (Summary)":
            st.header("Overall EV Battery Fleet Overview")
            st.markdown("This section provides a high-level overview of the entire second-life EV battery fleet based on charging data analysis.")

            # --- Fleet Health KPIs at a glance ---
            st.subheader("Fleet Health Summary (based on User-Defined Thresholds)")
            col_good, col_marginal, col_bad = st.columns(3)
            
            good_count = df_dashboard_data[df_dashboard_data["Health Category (User-Defined)"] == "Good"].shape[0]
            marginal_count = df_dashboard_data[df_dashboard_data["Health Category (User-Defined)"] == "Marginal"].shape[0]
            bad_count = df_dashboard_data[df_dashboard_data["Health Category (User-Defined)"] == "Bad"].shape[0]

            total_batteries = df_dashboard_data.shape[0]

            with col_good:
                st.metric("Good Batteries", f"{good_count} / {total_batteries} {get_battery_icon('Good')}")
            with col_marginal:
                st.metric("Marginal Batteries", f"{marginal_count} / {total_batteries} {get_battery_icon('Marginal')}")
            with col_bad:
                st.metric("Bad Batteries", f"{bad_count} / {total_batteries} {get_battery_icon('Bad')}")
            
            st.markdown("---")

            # --- Summary Statistics for Each Category ---
            st.subheader("Average Metrics by Health Category")
            if not df_dashboard_data.empty:
                # Dynamically include Avg SOC and Avg Degradation if columns exist
                agg_cols = {
                    'Avg_Capacity':('Capacity (Ah)', 'mean'),
                    'Avg_Voltage':('Avg Voltage', 'mean'),
                    'Avg_Temp':('Avg Temp', 'mean')
                }
                if 'Avg SOC (%)' in df_dashboard_data.columns:
                    agg_cols['Avg_SOC'] = ('Avg SOC (%)', 'mean')
                if 'Avg Degradation Rate (%)' in df_dashboard_data.columns:
                    agg_cols['Avg_Degradation'] = ('Avg Degradation Rate (%)', 'mean')

                avg_metrics_by_category = df_dashboard_data.groupby("Health Category (User-Defined)").agg(**agg_cols).round(2)
                st.dataframe(avg_metrics_by_category, use_container_width=True)
            else:
                st.info("No data to display average metrics by category.")

            st.markdown("---")

            # --- Classified Batteries Table with Filter and Sorting ---
            st.subheader("Classified Batteries Overview")
            st.markdown("View batteries categorized by their health status. You can filter and sort the table.")
            
            classification_source_filter = st.radio(
                "Filter by Classification Source:",
                ("User-Defined Thresholds", "KMeans Clustering"),
                key="classification_source_filter",
                horizontal=True
            )

            if classification_source_filter == "User-Defined Thresholds":
                category_column_to_filter = "Health Category (User-Defined)"
            else:
                category_column_to_filter = "Health Category (KMeans)"
            
            available_categories = sorted(df_dashboard_data[category_column_to_filter].unique().tolist())
            available_categories = [cat for cat in available_categories if cat not in ["N/A", "Not enough data", "Clustering Failed", "Not enough data variation"]]
            
            display_order = ["Good", "Marginal", "Bad", "Unknown"]
            sorted_available_categories = sorted(available_categories, key=lambda x: display_order.index(x) if x in display_order else len(display_order))


            selected_categories = st.multiselect(
                "Select Health Categories to Display:",
                options=sorted_available_categories,
                default=sorted_available_categories,
                key="category_filter"
            )

            filtered_df = df_dashboard_data[df_dashboard_data[category_column_to_filter].isin(selected_categories)]

            # --- Dynamic columns for table display ---
            display_cols_table = ['File Name', 'Avg Voltage', 'Avg Temp', 'Capacity (Ah)']
            if 'Avg SOC (%)' in df_dashboard_data.columns:
                display_cols_table.append('Avg SOC (%)')
            if 'Avg Degradation Rate (%)' in df_dashboard_data.columns:
                display_cols_table.append('Avg Degradation Rate (%)')
            display_cols_table.extend(['Health Category (KMeans)', 'Health Category (User-Defined)', 'Is Anomaly'])

            # --- Sorting options for the table ---
            sort_options_table = ['File Name', 'Avg Voltage', 'Avg Temp', 'Capacity (Ah)']
            if 'Avg SOC (%)' in df_dashboard_data.columns:
                sort_options_table.append('Avg SOC (%)')
            if 'Avg Degradation Rate (%)' in df_dashboard_data.columns:
                sort_options_table.append('Avg Degradation Rate (%)')
            sort_options_table.append('Health Category (User-Defined)') # Always include this for sorting by category

            sort_column = st.selectbox(
                "Sort table by:",
                options=sort_options_table,
                key="table_sort_column"
            )
            sort_ascending = st.checkbox("Sort Ascending", value=True, key="table_sort_ascending")
            
            if not filtered_df.empty:
                filtered_df_sorted = filtered_df.sort_values(by=sort_column, ascending=sort_ascending)
                st.dataframe(filtered_df_sorted[display_cols_table], use_container_width=True)
                
                if len(filtered_df_sorted) > 0:
                    st.markdown("---")
                    st.info("Select a battery below to view its detailed time-series data.")
                    quick_select_filtered = st.selectbox(
                        "Quick Select Battery from Filtered List:",
                        options=[""] + filtered_df_sorted["File Name"].tolist(),
                        key="quick_select_filtered_battery"
                    )
                    if quick_select_filtered:
                        # FIXED: Set the pre-selected session state variable and rerun
                        st.session_state.pre_selected_sidebar_battery = quick_select_filtered
                        st.rerun() 
            else:
                st.info(f"No batteries found in the selected categories for '{classification_source_filter}'.")


            st.markdown("---")
            with st.expander("View Full Battery Data Summary Table", expanded=False):
                st.subheader("📊 All Battery Data Summary")
                st.dataframe(df_dashboard_data[display_cols_table], use_container_width=True) # Use dynamic columns here too

            # --- Interactive Overall Charts Section ---
            st.subheader("📈 Fleet-wide Visualizations")

            st.markdown("#### Capacity Trend Across All Batteries")
            df_sorted_by_id = df_dashboard_data.sort_values(by="File Name").reset_index(drop=True)
            fig_capacity_trend = px.line(
                df_sorted_by_id, 
                x="File Name", 
                y="Capacity (Ah)", 
                title="Battery Capacity Over Different Models",
                markers=True,
                hover_data={"File Name": True, "Capacity (Ah)": ":.2f", 
                            "Health Category (KMeans)": True, 
                            "Health Category (User-Defined)": True,
                            "Is Anomaly": True,
                            # Dynamically add SOC/Degradation to hover if available
                            **({'Avg SOC (%)': ':.2f'} if 'Avg SOC (%)' in df_dashboard_data.columns else {}),
                            **({'Avg Degradation Rate (%)': ':.2f'} if 'Avg Degradation Rate (%)' in df_dashboard_data.columns else {})}
            )
            fig_capacity_trend.update_layout(xaxis_title="Battery File", yaxis_title="Capacity (Ah)")
            st.plotly_chart(fig_capacity_trend, use_container_width=True)

            st.markdown("#### Capacity vs. Average Voltage by Health Category")
            
            health_category_choice = st.radio(
                "Choose Health Category Source for Scatter Plot:",
                ("User-Defined Thresholds", "KMeans Clustering"),
                key="scatter_health_source"
            )
            
            color_column = "Health Category (User-Defined)" if health_category_choice == "User-Defined Thresholds" else "Health Category (KMeans)"

            fig_scatter = px.scatter(
                df_dashboard_data,
                x="Capacity (Ah)",
                y="Avg Voltage",
                color=color_column,
                color_discrete_map=health_color_map, # Apply custom colors
                title=f"Battery Health: Capacity vs. Average Voltage ({health_category_choice})",
                hover_data={"File Name": True, "Avg Temp": ":.2f", "Is Anomaly": True, 
                            # Dynamically add SOC/Degradation to hover if available
                            **({'Avg SOC (%)': ':.2f'} if 'Avg SOC (%)' in df_dashboard_data.columns else {}),
                            **({'Avg Degradation Rate (%)': ':.2f'} if 'Avg Degradation Rate (%)' in df_dashboard_data.columns else {})},
                size_max=15
            )
            fig_scatter.update_layout(xaxis_title="Capacity (Ah)", yaxis_title="Avg Voltage")
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("#### Distribution of Battery Health Categories")
            
            pie_health_category_choice = st.radio(
                "Choose Health Category Source for Pie Chart:",
                ("User-Defined Thresholds", "KMeans Clustering"),
                key="pie_health_source"
            )
            
            pie_column = "Health Category (User-Defined)" if pie_health_category_choice == "User-Defined Thresholds" else "Health Category (KMeans)"

            if pie_column in df_dashboard_data.columns and not df_dashboard_data[pie_column].isin(["N/A", "Not enough data", "Clustering Failed", "Not enough data variation"]).all():
                pie_data = df_dashboard_data[~df_dashboard_data[pie_column].isin(["N/A", "Not enough data", "Clustering Failed", "Not enough data variation"])][pie_column].value_counts().reset_index()
                pie_data.columns = ['Health Category', 'Count']

                if not pie_data.empty:
                    fig_pie = px.pie(
                        pie_data, 
                        values='Count', 
                        names='Health Category', 
                        title=f'Overall Battery Health Distribution ({pie_health_category_choice})',
                        hole=0.3,
                        color='Health Category', # Ensure color mapping works
                        color_discrete_map=health_color_map # Apply custom colors
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No valid health categories to display in the pie chart.")
            else:
                st.info("Health category distribution cannot be displayed due to insufficient or problematic data for clustering.")
            
            st.markdown("---")
            st.subheader("About This Project Dashboard")
            st.markdown("""
            This dashboard is a key component of the **"Second-Life EV Batteries: A Screening and Evaluation Framework"** project.
            
            **Its role in the project:**
            * **Python-based Screening:** It implements the core algorithm to categorize second-life EV batteries based on their measured health parameters (Voltage, Temperature, Capacity, Degradation Rate).
            * **Interactive Decision Support:** It allows users to define custom health thresholds, providing a flexible tool for assessing battery suitability for repurposing.
            * **Data Visualization & Analysis:** It offers fleet-wide overviews and detailed individual battery performance insights, crucial for identifying high-potential units.
            * **Performance Monitoring:** Displays data loading and processing times.
            
            This tool aims to enhance the efficiency and transparency of second-life battery deployment in stationary energy storage systems, contributing to sustainable energy infrastructure and circular economy principles.
            """)
            
            # --- Performance Monitoring Display ---
            st.markdown("---")
            st.subheader("Dashboard Performance")
            st.info(f"Data Last Refreshed: {st.session_state.get('last_refresh_time', 'N/A')}")
            st.info(f"Initial Data Processing Time: {st.session_state.get('processing_time', 'N/A')} seconds")


        elif selected_file == "Multi-Battery Comparison":
            st.header("⚖️ Multi-Battery Performance Comparison")
            st.markdown("Select multiple batteries to compare their time-series data side-by-side.")

            batteries_to_compare = st.multiselect(
                "Select batteries for comparison:",
                options=sorted(df_dashboard_data["File Name"].tolist()),
                default=df_dashboard_data["File Name"].head(2).tolist(),
                key="multi_compare_select"
            )

            if len(batteries_to_compare) > 0:
                # Get summary data for selected batteries for comparison logic
                df_selected_summary = df_dashboard_data[df_dashboard_data["File Name"].isin(batteries_to_compare)].copy()

                # --- BEST BATTERY SUGGESTION LOGIC ---
                if not df_selected_summary.empty:
                    # Sort by Health Category (User-Defined: Good < Marginal < Bad), then Capacity (desc), then Degradation Rate (asc)
                    df_selected_summary['Sort_Key'] = df_selected_summary['Health Category (User-Defined)'].apply(
                        lambda x: 0 if x == 'Good' else (1 if x == 'Marginal' else 2 if x == 'Bad' else 3)
                    )
                    sort_cols_metrics = ['Capacity (Ah)']
                    sort_asc_metrics = [False]
                    if 'Avg Degradation Rate (%)' in df_selected_summary.columns:
                        sort_cols_metrics.append('Avg Degradation Rate (%)')
                        sort_asc_metrics.append(True)
                    
                    df_selected_summary_ranked = df_selected_summary.sort_values(
                        by=['Sort_Key'] + sort_cols_metrics, ascending=[True] + sort_asc_metrics
                    ).reset_index(drop=True)
                    
                    best_battery = df_selected_summary_ranked.iloc[0]
                    
                    st.subheader("🏆 Best Battery Suggestion:")
                    st.success(f"Based on your selected batteries, **{best_battery['File Name']}** appears to be the most promising for efficient energy provision.")
                    st.markdown(f"**Key Metrics:** Capacity: {best_battery['Capacity (Ah)']:.2f} Ah, Avg Voltage: {best_battery['Avg Voltage']:.2f} V, Avg Temp: {best_battery['Avg Temp']:.2f} °C, User-Defined Health: **{best_battery['Health Category (User-Defined)']}**.")
                    if 'Avg Degradation Rate (%)' in best_battery and not pd.isna(best_battery['Avg Degradation Rate (%)']):
                        st.markdown(f"Avg Degradation Rate: {best_battery['Avg Degradation Rate (%)']:.2f} %.")
                    st.markdown("---")

                comparison_data = pd.DataFrame()
                for bat_file in batteries_to_compare:
                    df_temp = load_single_battery_data(bat_file, DATA_DIRECTORY)
                    if not df_temp.empty:
                        df_temp['Selected Battery File'] = bat_file
                        comparison_data = pd.concat([comparison_data, df_temp], ignore_index=True)
                
                if not comparison_data.empty:
                    st.markdown("#### Comparative Time-Series Data")
                    
                    fig_comp_voltage = px.line(
                        comparison_data, 
                        x="Time", 
                        y="Voltage_measured", 
                        color="Selected Battery File",
                        title="Comparative Voltage vs. Time",
                        hover_data={"Time": True, "Voltage_measured": ":.3f", 
                                    **({'SOC_Percent': ':.2f'} if 'SOC_Percent' in comparison_data.columns else {})}
                    )
                    fig_comp_voltage.update_layout(xaxis_title="Time (s)", yaxis_title="Voltage (V)")
                    st.plotly_chart(fig_comp_voltage, use_container_width=True)

                    fig_comp_current = px.line(
                        comparison_data, 
                        x="Time", 
                        y="Current_measured", 
                        color="Selected Battery File",
                        title="Comparative Current vs. Time",
                        hover_data={"Time": True, "Current_measured": ":.3f", 
                                    **({'SOC_Percent': ':.2f'} if 'SOC_Percent' in comparison_data.columns else {})}
                    )
                    fig_comp_current.update_layout(xaxis_title="Time (s)", yaxis_title="Current (A)")
                    st.plotly_chart(fig_comp_current, use_container_width=True)

                    fig_comp_temp = px.line(
                        comparison_data, 
                        x="Time", 
                        y="Temperature_measured", 
                        color="Selected Battery File",
                        title="Comparative Temperature vs. Time",
                        hover_data={"Time": True, "Temperature_measured": ":.3f", 
                                    **({'SOC_Percent': ':.2f'} if 'SOC_Percent' in comparison_data.columns else {})}
                    )
                    fig_comp_temp.update_layout(xaxis_title="Time (s)", yaxis_title="Temperature (°C)")
                    st.plotly_chart(fig_comp_temp, use_container_width=True)

                    if "SOC_Percent" in comparison_data.columns:
                        fig_comp_soc = px.line(
                            comparison_data, 
                            x="Time", 
                            y="SOC_Percent", 
                            color="Selected Battery File",
                            title="Comparative State of Charge (SOC) vs. Time",
                            hover_data={"Time": True, "SOC_Percent": ":.2f"}
                        )
                        fig_comp_soc.update_layout(xaxis_title="Time (s)", yaxis_title="SOC (%)")
                        st.plotly_chart(fig_comp_soc, use_container_width=True)

                else:
                    st.warning("No data loaded for selected batteries for comparison.")
            else:
                st.info("Please select at least one battery to compare.")

        else: # Detailed Analysis for a specific battery File Name
            st.header(f"🔍 Detailed Analysis for **{selected_file}**")
            st.markdown("Dive into the specific performance metrics and time-series data for this battery.")
            
            selected_battery_summary = df_dashboard_data[df_dashboard_data["File Name"] == selected_file].iloc[0]
            
            st.markdown("#### Key Performance Indicators")
            # Dynamically adjust KPI columns based on available data
            kpi_cols_to_show = 5 # Base for V, T, Cap, KMeans, User-Defined
            if 'Avg SOC (%)' in df_dashboard_data.columns: kpi_cols_to_show += 1
            if 'Avg Degradation Rate (%)' in df_dashboard_data.columns: kpi_cols_to_show += 1

            kpi_cols = st.columns(kpi_cols_to_show)
            
            # Populate KPIs dynamically
            kpi_idx = 0
            with kpi_cols[kpi_idx]:
                st.metric("Avg Voltage", f"{selected_battery_summary['Avg Voltage']:.2f} V")
            kpi_idx += 1
            with kpi_cols[kpi_idx]:
                st.metric("Avg Temp", f"{selected_battery_summary['Avg Temp']:.2f} °C")
            kpi_idx += 1
            with kpi_cols[kpi_idx]:
                st.metric("Capacity", f"{selected_battery_summary['Capacity (Ah)']:.2f} Ah")
            kpi_idx += 1
            if 'Avg SOC (%)' in df_dashboard_data.columns:
                with kpi_cols[kpi_idx]:
                    st.metric("Avg SOC", f"{selected_battery_summary['Avg SOC (%)']:.2f} %")
                kpi_idx += 1
            if 'Avg Degradation Rate (%)' in df_dashboard_data.columns:
                with kpi_cols[kpi_idx]:
                    st.metric("Avg Degradation", f"{selected_battery_summary['Avg Degradation Rate (%)']:.2f} %")
                kpi_idx += 1
            
            with kpi_cols[kpi_idx]:
                kmeans_health_cat = selected_battery_summary['Health Category (KMeans)']
                st.metric(f"KMeans Status {get_battery_icon(kmeans_health_cat)}", kmeans_health_cat)
            kpi_idx += 1
            with kpi_cols[kpi_idx]:
                user_health_cat = selected_battery_summary['Health Category (User-Defined)']
                st.metric(f"User-Defined Status {get_battery_icon(user_health_cat)}", user_health_cat)
            
            st.markdown(f"**Anomaly Status:** {'🚨 Anomaly Detected!' if selected_battery_summary['Is Anomaly'] else '✅ No Anomaly'}")

            # --- Battery Recommendation ---
            st.markdown("---")
            st.subheader("Battery Recommendation")
            user_defined_status = selected_battery_summary['Health Category (User-Defined)']
            recommendation_text = ""
            if user_defined_status == "Good":
                recommendation_text = "This battery shows **high potential for efficient energy provision** and is suitable for repurposing in stationary energy storage systems."
                st.success(f"✅ **Recommendation:** {recommendation_text}")
            elif user_defined_status == "Marginal":
                recommendation_text = "This battery has **marginal potential**. It may require further detailed testing or could be suitable for less demanding applications."
                st.warning(f"⚠️ **Recommendation:** {recommendation_text}")
            else: # Bad
                recommendation_text = "This battery is classified as **low potential**. It is likely unsuitable for repurposing and should be considered for recycling."
                st.error(f"❌ **Recommendation:** {recommendation_text}")
            
            # --- Techno-Economic Implications (Conceptual) ---
            st.markdown("---")
            st.subheader("Conceptual Techno-Economic Implications")
            st.markdown("""
            *This section conceptually links the battery's health status to its economic viability for second-life applications, as part of the Techno-Economic Analysis pillar of your project.*
            """)
            if user_defined_status == "Good":
                st.info("Based on its 'Good' health, this battery likely has a **high economic value** for repurposing, potentially leading to a low Levelized Cost of Storage (LCOS) and positive Net Present Value (NPV) in a grid-scale system.")
                st.markdown("**What-If Scenario:**")
                col_lcos_1, col_lcos_2 = st.columns(2)
                with col_lcos_1:
                    expected_duration = st.number_input("Expected Second-Life Duration (Years)", min_value=1, max_value=20, value=7, key=f"duration_good_{selected_file}") # Unique key
                with col_lcos_2:
                    energy_price = st.number_input("Assumed Energy Price (£/kWh)", min_value=0.05, max_value=0.50, value=0.15, step=0.01, format="%.2f", key=f"price_good_{selected_file}") # Unique key
                
                # Simple conceptual formulas for LCOS and NPV
                simulated_lcos = 0.10 - (selected_battery_summary['Capacity (Ah)'] * 0.001) + (expected_duration * 0.001) 
                simulated_npv = 1000 + (selected_battery_summary['Capacity (Ah)'] * 50) - (expected_duration * 10) 
                st.markdown(f"**Simulated LCOS:** £{simulated_lcos:.2f}/kWh (Lower is better)")
                st.markdown(f"**Simulated NPV:** £{simulated_npv:.2f} (Higher is better)")
                st.caption("Note: These are conceptual values. A full techno-economic model would be more complex.")

            elif user_defined_status == "Marginal":
                st.warning("With 'Marginal' health, the economic viability of this battery for repurposing might be **moderate**. Further testing is recommended to assess its true LCOS and NPV, or it might be suited for less critical applications.")
                st.markdown("**What-If Scenario:**")
                col_lcos_1, col_lcos_2 = st.columns(2)
                with col_lcos_1:
                    expected_duration = st.number_input("Expected Second-Life Duration (Years)", min_value=1, max_value=10, value=3, key=f"duration_marginal_{selected_file}") # Unique key
                with col_lcos_2:
                    energy_price = st.number_input("Assumed Energy Price (£/kWh)", min_value=0.05, max_value=0.50, value=0.10, step=0.01, format="%.2f", key=f"price_marginal_{selected_file}") # Unique key
                
                simulated_lcos = 0.25 - (selected_battery_summary['Capacity (Ah)'] * 0.001) + (expected_duration * 0.005)
                simulated_npv = 500 + (selected_battery_summary['Capacity (Ah)'] * 20) - (expected_duration * 20)
                st.markdown(f"**Simulated LCOS:** £{simulated_lcos:.2f}/kWh")
                st.markdown(f"**Simulated NPV:** £{simulated_npv:.2f}")
                st.caption("Note: These are conceptual values. A full techno-economic model would be more complex.")

            else: # Bad
                st.error("A 'Bad' health classification suggests **low economic viability** for repurposing. Its LCOS would likely be high, and NPV negative. It is primarily a candidate for material recovery through recycling.")
                st.markdown("**What-If Scenario:**")
                st.info("Repurposing this battery is likely not economically viable. Focus on recycling value.")
            
            st.markdown("---")

            df_single_battery = load_single_battery_data(selected_file, DATA_DIRECTORY)

            if not df_single_battery.empty:
                st.markdown("#### Time-Series Data for Selected Battery")
                
                ts_col1, ts_col2 = st.columns(2)

                with ts_col1:
                    fig_voltage = px.line(
                        df_single_battery, 
                        x="Time", 
                        y="Voltage_measured", 
                        title="Voltage vs. Time",
                        hover_data={"Time": True, "Voltage_measured": ":.3f", 
                                    **({'SOC_Percent': ':.2f'} if 'SOC_Percent' in df_single_battery.columns else {})}
                    )
                    fig_voltage.update_layout(xaxis_title="Time (s)", yaxis_title="Voltage (V)")
                    st.plotly_chart(fig_voltage, use_container_width=True)

                with ts_col2:
                    fig_current = px.line(
                        df_single_battery, 
                        x="Time", 
                        y="Current_measured", 
                        title="Current vs. Time",
                        hover_data={"Time": True, "Current_measured": ":.3f", 
                                    **({'SOC_Percent': ':.2f'} if 'SOC_Percent' in df_single_battery.columns else {})}
                    )
                    fig_current.update_layout(xaxis_title="Time (s)", yaxis_title="Current (A)")
                    st.plotly_chart(fig_current, use_container_width=True)
                
                st.markdown("---")
                ts_col3, ts_col4 = st.columns(2)
                with ts_col3:
                    fig_temp = px.line(
                        df_single_battery, 
                        x="Time", 
                        y="Temperature_measured", 
                        title="Temperature vs. Time",
                        hover_data={"Time": True, "Temperature_measured": ":.3f", 
                                    **({'SOC_Percent': ':.2f'} if 'SOC_Percent' in df_single_battery.columns else {})}
                    )
                    fig_temp.update_layout(xaxis_title="Time (s)", yaxis_title="Temperature (°C)")
                    st.plotly_chart(fig_temp, use_container_width=True)
                
                with ts_col4:
                    if "SOC_Percent" in df_single_battery.columns:
                        fig_soc = px.line(
                            df_single_battery, 
                            x="Time", 
                            y="SOC_Percent", 
                            title="State of Charge (SOC) vs. Time",
                            hover_data={"Time": True, "SOC_Percent": ":.2f"}
                        )
                        fig_soc.update_layout(xaxis_title="Time (s)", yaxis_title="SOC (%)")
                        st.plotly_chart(fig_soc, use_container_width=True)
                    else:
                        st.info("SOC (%) data not available for this battery.")

            else:
                st.warning("Detailed time-series data could not be loaded for this battery.")
        
        st.markdown("---")
        # --- Reporting/Exporting (Beyond CSV) ---
        st.subheader("Reporting & Export Options")
        st.download_button(
            label="📥 Download Summary Data as CSV",
            data=df_dashboard_data.to_csv(index=False),
            file_name="battery_health_summary.csv",
            mime="text/csv"
        )
        
        # Generate PDF Report
        # Determine which battery to report on for PDF
        pdf_df_summary = df_dashboard_data
        pdf_selected_battery_summary = None
        
        if selected_file != "All Batteries (Summary)" and selected_file != "Multi-Battery Comparison":
            pdf_selected_battery_summary = df_dashboard_data[df_dashboard_data["File Name"] == selected_file].iloc[0]

        pdf_output_bytes = create_pdf_report(
            pdf_df_summary, 
            pdf_selected_battery_summary, 
            health_color_map, # Pass color map for potential future use in PDF (not used in current basic PDF)
            st.session_state.get("scatter_health_source", "User-Defined Thresholds"), # Pass current scatter choice
            st.session_state.get("pie_health_source", "User-Defined Thresholds") # Pass current pie choice
        )

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_output_bytes,
            file_name="battery_screening_report.pdf",
            mime="application/pdf",
            key="download_pdf_report"
        )
        st.info("Note: The PDF report provides a text-based summary. Interactive plots are best viewed in the dashboard.")


else:
    st.error(f"""
    **Data Not Found!**
    Please ensure your **CSV** files (`.csv`) are located in the same folder as this script.
    Current folder: `{os.path.abspath(DATA_DIRECTORY)}`
    """)
    st.info("Make sure each CSV file contains the following columns: 'Voltage_measured', 'Temperature_measured', 'Current_measured', 'Time'. Optionally, 'SOC_Percent' and 'Degradation_Rate_Percent' for enhanced features.")

