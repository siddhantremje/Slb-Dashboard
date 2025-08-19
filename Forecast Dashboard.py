import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time # For performance monitoring
from pathlib import Path

# --- Configuration Section ---
LOGO_FILE_NAME = "ntu_logo.png"

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="EV Battery Supply Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Add Custom CSS for a Bright, Modern Theme ---
st.markdown(
    """
    <style>
    /* Overall page background and text color */
    .stApp {
        background-color: #f0f2f6;
        color: #262730;
    }
    /* Sidebar styling */
    .css-1d3f820 {
        background-color: #ffffff;
        color: #262730;
    }
    /* Main content container padding */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* Style for the KPI metrics */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 0.75rem;
        padding: 1.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        text-align: center;
        transition: transform 0.2s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
    }
    [data-testid="stMetricLabel"] > div {
        color: #555555;
        font-weight: bold;
        font-size: 1rem;
    }
    [data-testid="stMetricValue"] {
        color: #008CBA;
        font-size: 2.5rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    [data-testid="stMetricDelta"] {
        color: #888888;
        font-weight: normal;
        font-size: 0.8rem;
    }
    /* Ensure markdown and headings are visible with a dark color */
    h1, h2, h3, h4, h5, h6, .stMarkdown {
        color: #262730;
    }
    /* Info box text color fix for light theme */
    div.stAlert > div:first-child {
        color: #333333;
    }
    /* Set color for the download button text */
    .stDownloadButton > button {
        color: white !important;
        background-color: #008CBA;
    }
    .stDownloadButton > button:hover {
        background-color: #007bb5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Helper Function for Logo ---
def get_logo_path(file_name):
    if Path(file_name).exists():
        return file_name
    return None

# --- Dashboard Title and Logo ---
title_col, logo_col = st.columns([10, 2])
with title_col:
    title_text = "UK EV Battery Supply Forecast"
    st.markdown(f'<h1>📊 {title_text}</h1>', unsafe_allow_html=True)

with logo_col:
    logo_path = get_logo_path(LOGO_FILE_NAME)
    if logo_path:
        st.image(logo_path, width=90)
    else:
        st.info("Logo file not found.")

st.markdown("""
This model forecasts the future supply of Second-Life EV (SLB) batteries in the UK
by simulating new EV sales, retirement rates, and battery degradation over time.
""")

# --- Model Assumptions & Inputs (Sidebar) ---
st.sidebar.header("Model Assumptions")
st.sidebar.markdown("Adjust the sliders to run 'what-if' scenarios.")

# --- Initial EV Sales Data (fictional for demonstration, but based on real trends) ---
historical_sales_data = {
    2010: 100, 2011: 500, 2012: 1500, 2013: 3000, 2014: 6000, 2015: 10000,
    2016: 20000, 2017: 30000, 2018: 60000, 2019: 100000, 2020: 180000,
    2021: 250000, 2022: 350000, 2023: 500000, 2024: 650000
}

current_year = 2025
forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Years)", min_value=5, max_value=25, value=15
)
forecast_end_year = current_year + forecast_horizon

sales_growth_rate = st.sidebar.slider(
    "Annual EV Sales Growth Rate (%)", min_value=5, max_value=50, value=25
)
avg_battery_lifespan = st.sidebar.slider(
    "Average EV Battery Lifespan (Years)", min_value=8, max_value=15, value=10
)
initial_capacity = st.sidebar.number_input(
    "Average New EV Battery Capacity (kWh)", min_value=50, max_value=150, value=75
)
eol_capacity_threshold = st.sidebar.slider(
    "End-of-Life (EOL) Capacity Threshold (%)", min_value=50, max_value=85, value=80
)

# --- Model Calculation Function ---
@st.cache_data
def run_forecast(historical_sales, horizon, growth_rate, lifespan, new_capacity, eol_threshold):
    """
    Simulates the supply of second-life batteries and their capacities over time.
    """
    forecast_start_year = max(historical_sales.keys()) + 1
    
    sales_series = pd.Series(historical_sales)
    
    for year in range(forecast_start_year, current_year + horizon + 1):
        last_year_sales = sales_series.iloc[-1]
        sales_series[year] = last_year_sales * (1 + growth_rate / 100.0)
    
    df_forecast = pd.DataFrame({
        'Year': sales_series.index,
        'New EV Sales': sales_series.values
    })
    
    df_forecast['Total EVs on Road (Simulated)'] = df_forecast['New EV Sales'].cumsum()
    
    df_forecast['SLB Supply (Units)'] = 0
    df_forecast['Total SLB Capacity (GWh)'] = 0.0

    for index, row in df_forecast.iterrows():
        retirement_year = int(row['Year'])
        sales_year = retirement_year - lifespan
        
        if sales_year in historical_sales:
            num_retired_batteries = historical_sales[sales_year]
            remaining_capacity = new_capacity * (eol_threshold / 100.0)
            total_capacity_gwh = (num_retired_batteries * remaining_capacity) / 1000
            
            df_forecast.loc[df_forecast['Year'] == retirement_year, 'SLB Supply (Units)'] = num_retired_batteries
            df_forecast.loc[df_forecast['Year'] == retirement_year, 'Total SLB Capacity (GWh)'] = total_capacity_gwh
    
    return df_forecast

# --- Display Model Results ---
st.subheader(f"Forecast for UK Second-Life EV Battery Supply ({current_year} - {forecast_end_year})")

forecast_df = run_forecast(
    historical_sales_data,
    forecast_horizon,
    sales_growth_rate,
    avg_battery_lifespan,
    initial_capacity,
    eol_capacity_threshold
)

# --- Summary KPIs ---
st.markdown("#### Key Forecast Metrics")

kpi_row1_col1, kpi_row1_col2 = st.columns(2)
kpi_row2_col1, kpi_row2_col2 = st.columns(2)

present_slb = forecast_df[forecast_df['Year'] == current_year]
present_slb_units = present_slb['SLB Supply (Units)'].sum()
present_slb_capacity = present_slb['Total SLB Capacity (GWh)'].sum()
total_evs_present = forecast_df[forecast_df['Year'] <= current_year]['New EV Sales'].sum()

short_term_horizon = min(5, forecast_horizon)
short_term_end_year = current_year + short_term_horizon
short_term_slb = forecast_df[(forecast_df['Year'] > current_year) & (forecast_df['Year'] <= short_term_end_year)]
short_term_slb_units = short_term_slb['SLB Supply (Units)'].sum()
short_term_slb_capacity = short_term_slb['Total SLB Capacity (GWh)'].sum()

long_term_horizon = min(15, forecast_horizon)
long_term_end_year = current_year + long_term_horizon
long_term_slb = forecast_df[(forecast_df['Year'] > current_year) & (forecast_df['Year'] <= long_term_end_year)]
long_term_slb_units = long_term_slb['SLB Supply (Units)'].sum()
long_term_slb_capacity = long_term_slb['Total SLB Capacity (GWh)'].sum()

with kpi_row1_col1:
    st.metric(f"SLB Supply in {current_year}", f"{present_slb_units:,.0f} units 🔄")
    st.caption(f"Est. Capacity: {present_slb_capacity:,.2f} GWh")

with kpi_row1_col2:
    st.metric(f"Total EVs on UK Roads ({current_year})", f"{total_evs_present:,.0f} 🚗")
    st.caption(f"Estimated from cumulative sales up to {current_year}.")

with kpi_row2_col1:
    st.metric(f"SLB Supply in Next {short_term_horizon} Years", f"{short_term_slb_units:,.0f} units 📈")
    st.caption(f"Est. Capacity: {short_term_slb_capacity:,.2f} GWh")

with kpi_row2_col2:
    st.metric(f"SLB Supply in Next {long_term_horizon} Years", f"{long_term_slb_units:,.0f} units 🚀")
    st.caption(f"Est. Capacity: {long_term_slb_capacity:,.2f} GWh")

st.markdown("---")

# --- Interactive UK Map with Fictional EV Data ---
st.markdown("#### Current EV Distribution Across UK Cities (Conceptual)")
st.markdown("This section shows a simulated distribution of the EV fleet across major UK cities.")

# Fictional EV distribution data (for demonstration purposes)
ev_cities_data = {
    'City': ['London', 'Birmingham', 'Manchester', 'Glasgow', 'Leeds', 'Liverpool', 'Bristol', 'Edinburgh', 'Sheffield', 'Cardiff',
             'Newcastle', 'Belfast', 'Nottingham', 'Southampton', 'Leicester', 'Brighton', 'Plymouth', 'Aberdeen', 'Stoke-on-Trent', 'Coventry',
             'Hull', 'Derby', 'Reading', 'Luton', 'Walsall', 'Sunderland', 'Telford', 'Blackburn', 'Cambridge', 'Oxford'],
    'lat': [51.5074, 52.4862, 53.4808, 55.8642, 53.8008, 53.4084, 51.4545, 55.9533, 53.3811, 51.4816,
            54.9783, 54.5973, 52.9548, 50.9097, 52.6369, 50.8225, 50.3755, 57.1497, 53.0027, 52.4068,
            53.7454, 52.9225, 51.4543, 51.8781, 52.5852, 54.9061, 52.6738, 53.7483, 52.2053, 51.7520],
    'lon': [-0.1278, -1.8904, -2.2426, -4.2518, -1.5491, -2.9916, -2.5879, -3.1883, -1.4701, -3.1791,
            -1.6178, -5.9301, -1.1581, -1.4043, -1.1398, -0.1372, -4.1427, -2.0943, -2.1794, -1.5197,
            -0.3364, -1.4766, -0.9781, -0.4184, -1.9806, -1.3815, -2.4468, -2.4828, 0.1218, -1.2577],
    'Number of EVs': [450000, 150000, 100000, 80000, 75000, 65000, 60000, 50000, 45000, 40000,
                      35000, 30000, 28000, 25000, 23000, 20000, 18000, 17000, 15000, 14000,
                      12000, 11000, 10000, 9000, 8000, 7500, 7000, 6500, 6000, 5500],
}
df_ev_cities = pd.DataFrame(ev_cities_data)

# Use columns for a side-by-side layout of the table and chart
map_col, table_col = st.columns([2, 1])

with map_col:
    fig_map = px.scatter_mapbox(df_ev_cities, 
        lat="lat", 
        lon="lon", 
        hover_name="City", 
        hover_data={"Number of EVs": True},
        color="Number of EVs",
        size="Number of EVs",
        color_continuous_scale=px.colors.sequential.Viridis,
        zoom=4,
        center={"lat": 54.0, "lon": -2.0},
        height=500
    )
    fig_map.update_layout(mapbox_style="carto-darkmatter")
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

with table_col:
    st.markdown("#### Distribution Table")
    # Sort the table by the number of EVs for better readability
    df_ev_cities_sorted = df_ev_cities.sort_values(by="Number of EVs", ascending=False)
    st.dataframe(df_ev_cities_sorted[['City', 'Number of EVs']].reset_index(drop=True))


st.markdown("---")


# --- NEW: City-specific breakdown ---
st.markdown("#### EV Type Breakdown by City")
selected_city = st.selectbox(
    "Select a city to see a breakdown of EV types:",
    options=[''] + df_ev_cities['City'].tolist(),
    key='city_select'
)

if selected_city:
    st.info(f"**Simulated breakdown for {selected_city}:**")
    
    # Fictional breakdown data (will vary by city for a more realistic feel)
    city_breakdown = {
        'London': {'Private Cars': 65, 'Business Vans': 25, 'Other Transport': 10},
        'Birmingham': {'Private Cars': 70, 'Business Vans': 20, 'Other Transport': 10},
        'Manchester': {'Private Cars': 60, 'Business Vans': 30, 'Other Transport': 10},
        'Glasgow': {'Private Cars': 75, 'Business Vans': 15, 'Other Transport': 10},
        'Leeds': {'Private Cars': 68, 'Business Vans': 22, 'Other Transport': 10},
        'Bristol': {'Private Cars': 80, 'Business Vans': 15, 'Other Transport': 5},
        'Cardiff': {'Private Cars': 72, 'Business Vans': 18, 'Other Transport': 10},
        'Newcastle': {'Private Cars': 55, 'Business Vans': 35, 'Other Transport': 10},
        'Belfast': {'Private Cars': 70, 'Business Vans': 20, 'Other Transport': 10},
        'Nottingham': {'Private Cars': 65, 'Business Vans': 25, 'Other Transport': 10},
        'Southampton': {'Private Cars': 75, 'Business Vans': 15, 'Other Transport': 10},
        'Leicester': {'Private Cars': 60, 'Business Vans': 30, 'Other Transport': 10},
        'Brighton': {'Private Cars': 85, 'Business Vans': 10, 'Other Transport': 5},
        'Plymouth': {'Private Cars': 80, 'Business Vans': 15, 'Other Transport': 5},
        'Aberdeen': {'Private Cars': 65, 'Business Vans': 25, 'Other Transport': 10},
        'Stoke-on-Trent': {'Private Cars': 70, 'Business Vans': 20, 'Other Transport': 10},
        'Coventry': {'Private Cars': 60, 'Business Vans': 30, 'Other Transport': 10},
        'Hull': {'Private Cars': 68, 'Business Vans': 22, 'Other Transport': 10},
        'Derby': {'Private Cars': 72, 'Business Vans': 18, 'Other Transport': 10},
        'Reading': {'Private Cars': 80, 'Business Vans': 15, 'Other Transport': 5},
        'Luton': {'Private Cars': 55, 'Business Vans': 35, 'Other Transport': 10},
        'Walsall': {'Private Cars': 65, 'Business Vans': 25, 'Other Transport': 10},
        'Sunderland': {'Private Cars': 70, 'Business Vans': 20, 'Other Transport': 10},
        'Telford': {'Private Cars': 75, 'Business Vans': 15, 'Other Transport': 10},
        'Blackburn': {'Private Cars': 68, 'Business Vans': 22, 'Other Transport': 10},
        'Cambridge': {'Private Cars': 85, 'Business Vans': 10, 'Other Transport': 5},
        'Oxford': {'Private Cars': 80, 'Business Vans': 15, 'Other Transport': 5},
        'Liverpool': {'Private Cars': 60, 'Business Vans': 30, 'Other Transport': 10},
    }
    
    # Create a DataFrame for the selected city's breakdown
    df_breakdown = pd.DataFrame(
        list(city_breakdown.get(selected_city, {}).items()), 
        columns=['EV Type', 'Percentage']
    )
    
    if not df_breakdown.empty:
        fig_breakdown = px.pie(
            df_breakdown, 
            values='Percentage', 
            names='EV Type', 
            title=f'EV Type Distribution in {selected_city}',
            hole=0.3,
            template="plotly_dark",
            color_discrete_map={'Private Cars':'#636EFA', 'Business Vans':'#EF553B', 'Other Transport':'#00CC96'}
        )
        st.plotly_chart(fig_breakdown, use_container_width=True)
    else:
        st.warning("No breakdown data available for this city.")
st.markdown("---")


# --- Historical Sales Bar Chart ---
st.markdown("#### Historical Sales Context")
st.markdown("This bar chart provides context for the forecast by showing historical new EV sales in the UK.")
historical_df = pd.DataFrame(
    list(historical_sales_data.items()),
    columns=['Year', 'New EV Sales']
)
fig_hist = px.bar(
    historical_df,
    x='Year',
    y='New EV Sales',
    title='Historical UK New EV Sales (Fictional Data)',
    labels={'New EV Sales': 'Number of Units'},
    template="plotly_dark"
)
st.plotly_chart(fig_hist, use_container_container_width=True)

# --- Interactive Plotly Line Chart ---
st.markdown("#### Forecast Visualisation")
st.markdown("This chart allows you to explore the relationship between EV sales and the resulting supply of second-life batteries.")

# Let the user select which metrics to plot
metric_options = ['New EV Sales', 'SLB Supply (Units)', 'Total EVs on Road (Simulated)', 'Total SLB Capacity (GWh)']
selected_metrics = st.multiselect(
    "Select metrics to visualize:",
    options=metric_options,
    default=['New EV Sales', 'SLB Supply (Units)']
)

if selected_metrics:
    fig = px.line(
        forecast_df,
        x='Year',
        y=selected_metrics,
        title="Key Forecast Metrics over Time",
        labels={'value': 'Number of Units', 'Year': 'Year', 'variable': 'Metric'},
        template="plotly_dark"
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please select at least one metric to visualize.")

st.markdown("---")
st.subheader("Raw Forecast Data")
st.dataframe(forecast_df, use_container_width=True)
st.markdown("---")

# --- Model Assumptions & Limitations ---
st.subheader("Model Assumptions & Limitations")
with st.expander("Expand to view model assumptions for a robust analysis.", expanded=False):
    st.markdown("""
    This model is built on several key assumptions to provide a clear forecast. For your research, it is crucial to acknowledge these limitations and perform sensitivity analysis by adjusting the sliders.
    
    **Key Assumptions:**
    - **Sales Growth:** The model assumes a constant annual growth rate for new EV sales. In reality, this rate will fluctuate based on economic factors, policy changes, and consumer demand.
    - **Average Lifespan:** A fixed average battery lifespan is used to determine the year of retirement. In practice, this lifespan can vary significantly due to usage patterns, climate, and maintenance.
    - **Constant Capacity:** The model assumes a fixed average capacity for all new EV batteries and a consistent EOL threshold. Actual batteries will have a range of capacities and degradation rates.
    - **Single Market Focus:** The model is specific to the UK market and does not account for international dynamics.
    
    **Implications for Techno-Economic Analysis:**
    The outputs of this model (the forecast of SLB units and total capacity) serve as the foundation for your economic analysis. The sensitivity analysis you can perform by adjusting the sliders directly demonstrates how changes in these assumptions will impact your LCOS and NPV calculations.
    """)

# --- Data Download ---
csv_forecast = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button(
    "📥 Download Forecast Data as CSV",
    csv_forecast,
    "uk_ev_supply_forecast.csv",
    "text/csv",
    key='download-csv'
)
