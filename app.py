import requests
import pandas as pd

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

if __name__ == "__main__":
    # Replace with your actual API endpoint URL
    api_endpoint_url = "https://www.canada.ca/content/dam/ircc/documents/json/ee_rounds_123_en.json"

    print(f"Fetching data from {api_endpoint_url}...")
    api_data = fetch_data_from_api(api_endpoint_url)

    if api_data:
        print("Data fetched successfully. Creating DataFrame...")
        dataframe = create_dataframe_from_data(api_data.get('rounds'))

        if not dataframe.empty:
            print("DataFrame created:")
            #print(dataframe.head()) # Print the first 5 rows of the DataFrame
            print(f"\nDataFrame shape: {dataframe.shape}")
        else:
            print("Could not create DataFrame from the fetched data.")
    else:
        print("Failed to fetch data from the API.")
    
    #rename the default dictionary keys to actual values of CVRS score ranges
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
    canadasite='https://www.canada.ca'
    # Rename the columns
    # The 'inplace=True' argument modifies the DataFrame directly.
    # If you want to create a new DataFrame with renamed columns,
    # use: df_renamed = df.rename(columns=column_rename_map)
    dataframe.rename(columns=column_rename_map, inplace=True)
    dataframe['drawSize'] = dataframe['drawSize'].str.replace(',', '').astype(int)
    print(dataframe.iloc[:,[0,2,3,4,5,6,9,10,11,12,13]].head())