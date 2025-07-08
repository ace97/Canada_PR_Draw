import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from scipy.stats import norm
import numpy as np
from sklearn.linear_model import LinearRegression # For trendline prediction

# --- 1. Data Loading and Preprocessing ---
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
    # Safely convert to string first, then replace, then to numeric
    df['drawSize'] = df['drawSize'].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    df['drawCRS'] = df['drawCRS'].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    # Convert CRS band columns to numeric, handling commas and ensuring int type
    crs_band_columns_from_map = [col for col in column_rename_map.values() if col not in ['drawSize', 'drawCRS', 'Total']]
    for col in crs_band_columns_from_map:
        if col in df.columns: # Check if column actually exists in the DataFrame
            df[col] = df[col].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

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
min_draw_size_overall = int(df['drawSize'].min())
max_draw_size_overall = int(df['drawSize'].max())
draw_size_range = st.sidebar.slider(
    "Select Draw Size Range",
    min_value=min_draw_size_overall,
    max_value=max_draw_size_overall,
    value=(min_draw_size_overall, max_draw_size_overall)
)

# Filter by Draw CRS
min_draw_crs_overall = int(df['drawCRS'].min())
max_draw_crs_overall = int(df['drawCRS'].max())
draw_crs_range = st.sidebar.slider(
    "Select Draw CRS Range",
    min_value=min_draw_crs_overall,
    max_value=max_draw_crs_overall,
    value=(min_draw_crs_overall, max_draw_crs_overall)
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

# --- 6. Basic Visualization (CRS Score over Time) ---
st.header("CRS Score Trend Over Time")
if not filtered_df.empty:
    # Sort by date for proper time series plotting
    chart_data = filtered_df.sort_values('drawDate')[['drawDate', 'drawCRS']]
    
    fig = px.line(chart_data, x='drawDate', y='drawCRS', 
                  title='CRS Cut-off Score Over Time',
                  labels={'drawDate': 'Draw Date', 'drawCRS': 'CRS Score'})
    
    # Add a trendline
    fig.update_traces(mode='lines+markers') # Ensure markers are visible if desired
    fig.update_layout(hovermode="x unified") # For better hovering
    
    # Check if there's enough data for a trendline
    if len(chart_data) > 1:
        fig.add_trace(px.scatter(chart_data, x='drawDate', y='drawCRS', trendline='ols').data[1]) # Add OLS trendline
    else:
        st.info("Not enough data points to display a trendline.")

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data to display CRS score trend.")

# --- 7. Probability Calculation and Distribution of Points ---
st.header("Probability of Being Invited in Next Draw")
st.markdown("Please note: These are statistical estimations based on your selected filters and should not be taken as a guarantee. Actual CRS cut-offs are unpredictable.")

if not filtered_df.empty:
    current_filtered_data = filtered_df.copy() # Work on a copy
    
    historical_crs_scores = current_filtered_data['drawCRS']
    average_draw_size = current_filtered_data['drawSize'].mean()


    st.subheader("1. Trend-Based Prediction (Based on Historical Cut-off Scores)")
    st.markdown("This prediction uses the historical variability of CRS cut-off scores from your filtered draws.")

    mean_crs = historical_crs_scores.mean()
    std_crs = historical_crs_scores.std()

    user_crs_score = st.slider(
        "Enter Your Current CRS Score for Analysis",
        min_value=int(filtered_df['drawCRS'].min()), # Use overall min/max for broader range
        max_value=int(filtered_df['drawCRS'].max()) + 50, # A bit higher than max to allow for higher scores
        value=int(mean_crs) if not np.isnan(mean_crs) else 450 # Default to mean or a reasonable value
    )

    if std_crs > 0:
        probability_trend = norm.cdf(user_crs_score, loc=mean_crs, scale=std_crs)
        
        st.markdown(f"With a CRS score of **{user_crs_score}**:")
        
        if user_crs_score >= historical_crs_scores.max():
            st.success(f"Your CRS score is **higher than or equal to the highest recorded CRS cut-off ({int(historical_crs_scores.max())}) from your selected draws**! This indicates a very high chance based on past trends.")
        elif user_crs_score <= historical_crs_scores.min():
            st.error(f"Your CRS score is **lower than or equal to the lowest recorded CRS cut-off ({int(historical_crs_scores.min())}) from your selected draws**. This indicates a very low chance based on past trends.")
        else:
            st.info(f"Based on the historical distribution of CRS scores from your selected draws (Mean: {mean_crs:.2f}, Std Dev: {std_crs:.2f}), there is an approximate **{probability_trend*100:.2f}%** chance that the next draw cut-off will be at or below your score, meaning you could receive an Invitation to Apply (ITA).")

        st.markdown(f"*(This means {probability_trend*100:.2f}% of historical draws had a CRS cut-off at or below your score.)*")
        
    else:
        st.info("Not enough variation in historical CRS scores from your selected draws to calculate a meaningful trend-based probability. All past selected draws had the same CRS cut-off.")
        if not np.isnan(mean_crs):
            st.info(f"The CRS cut-off for all past selected draws was **{int(mean_crs)}**.")
            if user_crs_score >= mean_crs:
                st.success(f"Your CRS score ({user_crs_score}) is at or above the consistent past draw cut-off. High chance!")
            else:
                st.error(f"Your CRS score ({user_crs_score}) is below the consistent past draw cut-off. Low chance.")

    st.markdown("---")
    st.subheader("2. Distribution-Based Prediction (Based on Latest Pool Composition)")
    st.markdown("This prediction estimates the cut-off by considering the number of candidates in each CRS score band in the latest pool.")

    # Define the CRS band columns from the map for robustness
    crs_band_columns = [
        "601-1200", "501-600", "451-500", "491-500", "481-490", "471-480",
        "461-470", "451-460", "401-450", "441-450", "431-440", "421-430",
        "411-420", "401-410", "351-400", "301-350", "0-300"
    ]
    
    # Find the latest draw date in the filtered data
    latest_draw_date = current_filtered_data['drawDate'].max()
    latest_draw_df = current_filtered_data[current_filtered_data['drawDate'] == latest_draw_date]

    latest_draw_row = pd.Series() # Initialize as empty Series
    
    if not latest_draw_df.empty:
        latest_draw_row = latest_draw_df.sort_values(by='drawNumber', ascending=False).iloc[0]
        existing_crs_band_columns = [col_name for col_name in crs_band_columns if col_name in latest_draw_row.index]
        
        if existing_crs_band_columns:
            crs_distribution_series = latest_draw_row[existing_crs_band_columns]
            
            # Create a DataFrame for plotting
            crs_distribution_df = pd.DataFrame({
                'CRS Band': crs_distribution_series.index,
                'Number of Candidates': crs_distribution_series.values
            })
            
            # For consistent sorting, extract the lower bound for sorting
            def get_lower_bound(band_str):
                try:
                    return int(band_str.split('-')[0])
                except:
                    return 0 # Default for edge cases like "0-300" or malformed
            
            crs_distribution_df['Sort_Key'] = crs_distribution_df['CRS Band'].apply(get_lower_bound)
            crs_distribution_df = crs_distribution_df.sort_values(by='Sort_Key', ascending=False).drop('Sort_Key', axis=1)

            st.write(f"Showing candidate distribution for the latest draw: **{latest_draw_row['drawDate'].strftime('%Y-%m-%d')} - {latest_draw_row['drawName']}**")
            fig_dist = px.bar(crs_distribution_df, 
                              x='CRS Band', 
                              y='Number of Candidates',
                              title='Distribution of Candidates in CRS Bands (Latest Filtered Draw)',
                              labels={'CRS Band': 'CRS Score Range', 'Number of Candidates': 'Total Candidates'})
            st.plotly_chart(fig_dist, use_container_width=True)
            
            total_candidates_in_pool = crs_distribution_series.sum()
            st.write(f"Estimated total candidates in the pool on this date: **{total_candidates_in_pool:,.0f}**")


            # Function to calculate estimated cut-off based on a given draw size
            def calculate_estimated_cut_off(target_draw_size, parsed_bands):
                cumulative_candidates_from_top = 0
                estimated_cut_off = None
                
                if target_draw_size > total_candidates_in_pool:
                    return 0 # Very low cut-off if draw size is greater than pool
                
                for band in parsed_bands:
                    candidates_in_current_band = band['candidates']
                    
                    if cumulative_candidates_from_top + candidates_in_current_band >= target_draw_size:
                        candidates_needed_from_this_band = target_draw_size - cumulative_candidates_from_top
                        
                        band_range = band['upper'] - band['lower'] + 1
                        if candidates_in_current_band > 0 and band_range > 0:
                            score_offset_from_top_of_band = (candidates_needed_from_this_band / candidates_in_current_band) * band_range
                            estimated_cut_off = band['upper'] - score_offset_from_top_of_band
                            estimated_cut_off = round(estimated_cut_off)
                        else:
                            estimated_cut_off = band['lower']
                        break
                    
                    cumulative_candidates_from_top += candidates_in_current_band
                
                return estimated_cut_off if estimated_cut_off is not None else parsed_bands[-1]['lower'] # Fallback

            # Parse CRS band ranges for calculation
            parsed_bands_data_for_calc = []
            for band_name in existing_crs_band_columns:
                lower = 0
                upper = 1200 # Default max CRS
                try:
                    if '-' in band_name:
                        parts = band_name.split('-')
                        lower = int(parts[0])
                        upper = int(parts[1])
                    elif band_name == '0-300':
                        lower = 0
                        upper = 300
                    elif band_name == '601-1200':
                        lower = 601
                        upper = 1200
                except ValueError:
                    continue
                num_candidates = crs_distribution_series.get(band_name, 0)
                parsed_bands_data_for_calc.append({'band_name': band_name, 'lower': lower, 'upper': upper, 'candidates': num_candidates})
            parsed_bands_data_for_calc.sort(key=lambda x: x['lower'], reverse=True) # Sort highest to lowest score

            st.markdown("##### 2a. Based on Average Draw Size")
            if average_draw_size > 0:
                estimated_cut_off_avg_draw = calculate_estimated_cut_off(average_draw_size, parsed_bands_data_for_calc)
                
                st.write(f"- Using average invitations issued per draw from selected filters: **{average_draw_size:,.0f}**")
                st.metric(label="Estimated CRS Cut-off (Avg Draw Size)", value=int(estimated_cut_off_avg_draw))

                if user_crs_score >= estimated_cut_off_avg_draw:
                    st.success(f"Your CRS score of **{user_crs_score}** is **at or above** this estimated cut-off. High chance based on average draw sizes and current pool.")
                else:
                    st.info(f"Your CRS score of **{user_crs_score}** is **below** this estimated cut-off. You would likely need the cut-off to drop or a larger draw.")
            else:
                st.info("Average draw size is zero or not available for your filtered selection, cannot perform this calculation.")

            st.markdown("##### 2b. Based on User-Provided Draw Size")
            user_provided_draw_size = st.number_input(
                "Enter a Custom Draw Size for Prediction:",
                min_value=1,
                max_value=total_candidates_in_pool + 5000, # Allow for slightly larger than current pool for simulation
                value=int(average_draw_size) if not np.isnan(average_draw_size) else 1000, # Default to average or 1000
                step=100
            )

            if user_provided_draw_size > 0:
                estimated_cut_off_user_draw = calculate_estimated_cut_off(user_provided_draw_size, parsed_bands_data_for_calc)
                
                st.metric(label=f"Estimated CRS Cut-off (for {user_provided_draw_size:,.0f} ITAs)", value=int(estimated_cut_off_user_draw))

                if user_crs_score >= estimated_cut_off_user_draw:
                    st.success(f"For a draw of **{user_provided_draw_size:,.0f}** ITAs, your CRS score of **{user_crs_score}** suggests a **high chance** of being invited.")
                else:
                    st.info(f"For a draw of **{user_provided_draw_size:,.0f}** ITAs, your CRS score of **{user_crs_score}** is below the estimated cut-off. You would need a larger draw or a lower cut-off.")
            else:
                st.info("Please enter a valid draw size greater than zero for this prediction.")

        else:
            st.info("CRS band distribution data not available for the latest filtered draw to perform distribution-based prediction.")
    else:
        st.info("No latest draw found with CRS band distribution data in the filtered set to perform distribution-based prediction.")

    st.markdown("---")
    st.subheader("Summary of Predictions")
    
    # Check if trend-based prediction was possible
    trend_prediction_possible = (std_crs is not None and std_crs > 0)
    
    # Check if distribution-based prediction (average) was possible
    dist_avg_prediction_possible = (latest_draw_row is not None and not latest_draw_row.empty and 'candidates' in parsed_bands_data_for_calc[0] and average_draw_size > 0)
    
    # Check if distribution-based prediction (user-provided) was possible
    dist_user_prediction_possible = (latest_draw_row is not None and not latest_draw_row.empty and 'candidates' in parsed_bands_data_for_calc[0] and user_provided_draw_size > 0)

    if trend_prediction_possible or dist_avg_prediction_possible or dist_user_prediction_possible:
        
        st.markdown(f"Your current CRS score: **{user_crs_score}**")
        st.markdown("---")

        if trend_prediction_possible:
            st.markdown(f"**Trend-Based Probability:** There is an approximate **{probability_trend*100:.2f}%** chance that the next draw cut-off will be at or below your score, based on historical cut-off variability.")
        else:
            st.markdown("**Trend-Based Probability:** Not calculable due to insufficient historical CRS score variation.")
        
        if dist_avg_prediction_possible and estimated_cut_off_avg_draw is not None:
            st.markdown(f"**Distribution-Based (Average Draw Size):** Estimated cut-off is **{int(estimated_cut_off_avg_draw)}** (for average draw size of {average_draw_size:,.0f} ITAs).")
            if user_crs_score >= estimated_cut_off_avg_draw:
                st.markdown("Outcome: **High chance** based on current pool and average draw size.")
            else:
                st.markdown("Outcome: **Lower chance** based on current pool and average draw size.")
        else:
            st.markdown("**Distribution-Based (Average Draw Size):** Not calculable or data insufficient.")
            
        if dist_user_prediction_possible and estimated_cut_off_user_draw is not None:
            st.markdown(f"**Distribution-Based (User-Provided Draw Size):** Estimated cut-off is **{int(estimated_cut_off_user_draw)}** (for your specified draw size of {user_provided_draw_size:,.0f} ITAs).")
            if user_crs_score >= estimated_cut_off_user_draw:
                st.markdown("Outcome: **High chance** for your specified draw size.")
            else:
                st.markdown("Outcome: **Lower chance** for your specified draw size.")
        else:
            st.markdown("**Distribution-Based (User-Provided Draw Size):** Not calculable or data insufficient.")

        st.markdown("---")
        st.info("Use these insights to understand your standing. Remember that IRCC draws are dynamic and influenced by many factors including policy changes and program-specific invitations.")

    else:
        st.info("No data available to generate predictions. Please adjust filters or check data availability for the latest draw and historical CRS scores.")

else:
    st.info("Apply filters or display raw data to enable probability calculations.")


# --- 8. Display Data Table (Moved to end) ---
st.markdown("---")
st.header("Raw Express Entry Draws Data")
st.markdown("*(This table displays the full data, or the filtered data if 'Show Raw Data' is unchecked in the sidebar.)*")
if not filtered_df.empty:
    st.dataframe(filtered_df[selected_columns], use_container_width=True)
else:
    st.warning("No data available for the selected filters.")


st.markdown("---")
st.caption("Data source: https://www.canada.ca/en/immigration-refugees-citizenship/corporate/mandate/policies-operational-instructions-agreements/ministerial-instructions/express-entry-rounds.html")