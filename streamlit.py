import streamlit as st
import pandas as pd
import requests

# --- 1. Data Loading and Preprocessing ---
# Removed @st.cache_data as per your request
def load_and_preprocess_data():
    """
    Loads the sample JSON data, creates a DataFrame, and preprocesses it.
    """
    api_endpoint_url = "https://www.canada.ca/content/dam/ircc/documents/json/ee_rounds_123_en.json"
    def fetch_data_from_api(api_url):
        """
        Fetches data from a given API endpoint.

        Args:
            api_url (str): The URL of the API endpoint.

        Returns:
            dict or list: The JSON response from the API, or None if an error occurs.
        """
        try:
            response = requests.get(api_url)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from API: {e}")
            return None

    def create_dataframe_from_data(data):
        """
        Creates a pandas DataFrame from the fetched data.

        Args:
            data (dict or list): The data fetched from the API.

        Returns:
            pandas.DataFrame: A DataFrame created from the data, or an empty DataFrame if data is None.
        """
        if data is None:
            return pd.DataFrame() # Return empty DataFrame if data is None
        try:
            # Assuming the data is a list of dictionaries or a dictionary that can be converted
            # You might need to adjust this based on the actual API response structure
            df = pd.DataFrame(data)
            return df
        except ValueError as e:
            print(f"Error creating DataFrame: {e}")
            return pd.DataFrame() # Return empty DataFrame on error

    api_data = fetch_data_from_api(api_endpoint_url)
    df = create_dataframe_from_data(api_data.get('rounds'))



    # Define the mapping for renaming columns
    column_rename_map = {
        "dd1": "601-1200",
        "dd2": "501-600",
        "dd3": "451-500",
        "dd4": "491-500",
        "dd5": "481-490",
        "dd6": "471-480",
        "dd7": "461-470",
        "dd8": "451-460",
        "dd9": "401-450",
        "dd10": "441-450",
        "dd11": "431-440",
        "dd12": "421-430",
        "dd13": "411-420",
        "dd14": "401-410",
        "dd15": "351-400",
        "dd16": "301-350",
        "dd17": "0-300",
        "dd18": "Total"
    }
    df.rename(columns=column_rename_map, inplace=True)

    # Convert 'drawDate' to datetime objects
    df['drawDate'] = pd.to_datetime(df['drawDate'])
    # Convert 'drawSize' and 'drawCRS' to numeric
    df['drawSize'] = df['drawSize'].str.replace(',', '').astype(int)
    df['drawCRS'] = df['drawCRS'].str.replace(',', '').astype(int)

    return df

# Load the data
df = load_and_preprocess_data()

# --- 2. Streamlit App Layout ---
st.set_page_config(
    page_title="Canada Express Entry Draws Dashboard",
    page_icon="🇨🇦",
    layout="wide"
)

st.title("🇨🇦 Canada Express Entry PR Draws Dashboard")
st.markdown("Explore and filter Express Entry invitation rounds data.")

# --- 3. Sidebar for Filters and Options ---
st.sidebar.header("Filter Options")

# Filter by Draw Name
all_draw_names = df['drawName'].unique().tolist()
selected_draw_names = st.sidebar.multiselect(
    "Select Draw Name(s)",
    options=all_draw_names,
    default=all_draw_names # Select all by default
)

# Filter by Draw Date Range
min_date = df['drawDate'].min().date()
max_date = df['drawDate'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Ensure date_range has two dates
if len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
elif len(date_range) == 1:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[0]) # If only one date selected, treat as a single day range
else: # Default to full range if nothing selected or invalid
    start_date = df['drawDate'].min()
    end_date = df['drawDate'].max()


# Filter by Draw Size
min_draw_size = int(df['drawSize'].min())
max_draw_size = int(df['drawSize'].max())
draw_size_range = st.sidebar.slider(
    "Select Draw Size Range",
    min_value=min_draw_size,
    max_value=max_draw_size,
    value=(min_draw_size, max_draw_size)
)

# Filter by Draw CRS
min_draw_crs = int(df['drawCRS'].min())
max_draw_crs = int(df['drawCRS'].max())
draw_crs_range = st.sidebar.slider(
    "Select Draw CRS Range",
    min_value=min_draw_crs,
    max_value=max_draw_crs,
    value=(min_draw_crs, max_draw_crs)
)

st.sidebar.markdown("---")
st.sidebar.header("Display Options")

# Toggle to show raw data vs. filtered data
show_raw_data = st.sidebar.checkbox("Show Raw Data (ignore filters)", value=False)

# Select columns to display
all_columns = df.columns.tolist()
# Exclude URL columns by default for cleaner display
default_display_columns = [col for col in all_columns if not ('URL' in col or 'Text' in col or 'mitext' in col)]
selected_columns = st.sidebar.multiselect(
    "Select Columns to Display",
    options=all_columns,
    default=default_display_columns
)

# --- 4. Apply Filters ---
if show_raw_data:
    filtered_df = df.copy()
    st.info("Displaying raw data. Filters are ignored.")
else:
    filtered_df = df[
        (df['drawName'].isin(selected_draw_names)) &
        (df['drawDate'] >= start_date) &
        (df['drawDate'] <= end_date) &
        (df['drawSize'] >= draw_size_range[0]) &
        (df['drawSize'] <= draw_size_range[1]) &
        (df['drawCRS'] >= draw_crs_range[0]) &
        (df['drawCRS'] <= draw_crs_range[1])
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

# --- 5. Display Key Metrics ---
st.header("Key Metrics")
if not filtered_df.empty:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Draws (Filtered)", value=len(filtered_df))
    with col2:
        st.metric(label="Average Draw Size (Filtered)", value=f"{filtered_df['drawSize'].mean():,.0f}")
    with col3:
        st.metric(label="Average CRS Score (Filtered)", value=f"{filtered_df['drawCRS'].mean():.2f}")
else:
    st.warning("No data matches the selected filters for metrics.")

# --- 6. Display Data Table ---
st.header("Express Entry Draws Data")
if not filtered_df.empty:
    st.dataframe(filtered_df[selected_columns], use_container_width=True)
else:
    st.warning("No data available for the selected filters.")

# --- 7. Basic Visualization (CRS Score over Time) ---
st.header("CRS Score Trend Over Time")
if not filtered_df.empty:
    # Sort by date for proper time series plotting
    chart_data = filtered_df.sort_values('drawDate')[['drawDate', 'drawCRS']]
    st.line_chart(chart_data.set_index('drawDate'))
else:
    st.info("No data to display CRS score trend.")

st.markdown("---")
st.caption("Data source: https://www.canada.ca/en/immigration-refugees-citizenship/corporate/mandate/policies-operational-instructions-agreements/ministerial-instructions/express-entry-rounds.html")
