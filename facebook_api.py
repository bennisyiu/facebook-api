import pandas as pd
import json
import requests
import pytz
import psycopg2
import os
from datetime import timedelta, datetime
from pandas import json_normalize
from dotenv import load_dotenv


def get_credentials():
    """Load and validate credentials from environment variables."""
    load_dotenv()  # Load .env file

    # Facebook credentials
    fb_token = os.getenv("FB_ACCESS_TOKEN")
    if not fb_token:
        raise ValueError("FB_ACCESS_TOKEN is missing in .env!")

    fb_credentials = {
        "access_token": fb_token,
        "ad_account_id": os.getenv("FB_AD_ACCOUNT_ID")
    }

    # Database credentials
    db_credentials = {
        "hostname": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT")),  # Convert to integer
        "username": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME"),
        "schema": os.getenv("DB_SCHEMA")
    }

    return fb_credentials, db_credentials


def fb_api_caller(url, params):
    """
    Generic Facebook API caller function to handle pagination and data retrieval.
    """
    all_data = []

    # Loop through pagination until no more pages
    while url:
        response = requests.get(url, params=params)

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            break

        if 'data' in data:
            all_data.extend(data['data'])  # Collect data from this page
        else:
            print("No data found in response")
            break

        # Check if there's a next page
        if 'paging' in data and 'next' in data['paging']:
            url = data['paging']['next']  # Follow the next link
            params = {}  # Clear params for subsequent requests since next link contains them
        else:
            url = None  # No more pages

    return all_data


def fb_insights_caller(fb_credentials, start_date, end_date, level):
    """
    Fetch Facebook insights data.
    """
    access_token = fb_credentials['access_token']
    ad_account_id = fb_credentials['ad_account_id']

    url = f'https://graph.facebook.com/v22.0/act_{ad_account_id}/insights'
    params = {
        'access_token': access_token,
        'time_range': json.dumps({'since': start_date, 'until': end_date}),
        'level': {level},
        'fields': 'campaign_name,outbound_clicks,spend,campaign_id,reach,actions,frequency,impressions',
        'time_increment': 1
    }

    # Fetch and return the data
    all_data = fb_api_caller(url, params)
    # print(all_data)

    # Pretty-print the data and let JSON handle Unicode properly
    # print(json.dumps(all_data, ensure_ascii=False, indent=4))
    print(
        f'{len(all_data)} of insights data from {start_date} to {end_date} will be inserted.')
    return all_data


def fb_status_caller(fb_credentials, start_date, end_date, level):
    """
    Fetch Facebook campaign status data.
    """
    access_token = fb_credentials['access_token']
    ad_account_id = fb_credentials['ad_account_id']

    url = f'https://graph.facebook.com/v21.0/act_{ad_account_id}/campaigns'
    params = {
        'access_token': access_token,
        'time_range': json.dumps({'since': start_date, 'until': end_date}),
        'level': {level},
        'fields': 'id,name,effective_status,status,start_time,stop_time',
        'time_increment': 1
    }

    # Fetch and return the data
    all_data = fb_api_caller(url, params)

    # Pretty-print the data and let JSON handle Unicode properly
    # print(json.dumps(all_data, ensure_ascii=False, indent=4))
    print(f'{len(all_data)} of status data from will be inserted.')

    return all_data


def process_fb_insights(all_data):
    """Process FB insights data extracted from API through fd_insights_caller()
    Return a dataframe with columns mapped: 
        'date','campaign_id','campaign_name',
        'spend', 'reach', 'link_click','frequency',
        'impressions', 'results','result_type','cost_per_result',
        'delivery_level','file_path','date_start',
        'date_stop', 'report_start_date', 'report_end_date','updated_at'
    """
    # Convert to DataFrame
    df = pd.json_normalize(all_data)

    # Extract link_click from actions
    if 'actions' in df.columns:
        df['link_click'] = df['actions'].apply(
            lambda actions: int(next(
                (action['value'] for action in actions
                 if isinstance(actions, list) and action['action_type'] == 'link_click'),
                0  # default value if link_click not found
            ))
        )

        # Extract results and result_type from actions
        df['results'] = df['actions'].apply(
            lambda actions: int(next(
                (action['value'] for action in actions
                 if isinstance(actions, list) and action['action_type'] == 'link_click'),
                0  # default value if no other action found
            ))
        )

        # Drop the original actions column
        df = df.drop('actions', axis=1)

    # Add result_type column (you might need to adjust this logic based on your specific needs)
    df['result_type'] = 'link_click'  # default value, adjust as needed

    # Convert spend to float first
    df['spend'] = pd.to_numeric(df['spend'], errors='coerce')

    # Calculate cost_per_result using the same logic as the original combine query
    df['results'] = df.apply(
        lambda row: row['results'] if row['results'] > row['link_click'] else row['link_click'],
        axis=1
    )

    # Calculate cost_per_result
    df['cost_per_result'] = df.apply(
        lambda row: 0 if row['results'] == 0 else row['spend'] /
        row['results'],
        axis=1
    )

    # default values
    df['date'] = df['date_start']
    df['report_start_date'] = df['date_start']
    df['report_end_date'] = df['date_stop']
    df['updated_at'] = datetime.now()
    df['delivery_level'] = 'campaign'
    df['file_path'] = 'Facebook API'

    # Drop all columns except the ones we want
    columns_to_keep = [
        'date',
        'campaign_id',
        'campaign_name',
        'spend',
        'reach',
        'link_click',
        'frequency',
        'impressions',
        'results',
        'result_type',
        'cost_per_result',
        'delivery_level',
        'file_path',
        'date_start',
        'date_stop',
        'report_start_date',
        'report_end_date',
        'updated_at'
    ]

    df = df[columns_to_keep]

    # Convert numeric columns
    numeric_columns = [
        'spend',
        'reach',
        'link_click',
        'frequency',
        'impressions',
        'results',
        'cost_per_result'
    ]

    # Convert all numeric columns
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert date columns
    date_columns = ['date_start', 'date_stop', 'date',
                    'report_start_date', 'report_end_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return df


def process_fb_status(all_data):
    """Process FB status data extracted from through fd_status_caller()
    Return a dataframe with columns mapped: 
        campaign_id, campaign_name, delivery_status, ad_start_time, ad_stop_time
    """
    # Convert to DataFrame
    df = pd.json_normalize(all_data)

    # Rename columns
    column_mapping = {
        'id': 'campaign_id',
        'name': 'campaign_name',
        'status': 'delivery_status',
        'start_time': 'ad_start_time',
        'stop_time': 'ad_stop_time'
    }

    df = df.rename(columns=column_mapping)

    # Keep only the columns we want
    columns_to_keep = [
        'campaign_id',
        'campaign_name',
        'delivery_status',
        'ad_start_time',
        'ad_stop_time'
    ]

    df = df[columns_to_keep]

    # Convert datetime columns and ensure UTC timezone
    datetime_columns = ['ad_start_time', 'ad_stop_time']
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)

    return df


def combine_fb_data(insights_df, status_df):
    """
    Combines Facebook insights and status DataFrames based on campaign_id.

    Args:
        insights_df: DataFrame from process_fb_insights
        status_df: DataFrame from process_fb_status

    Returns:
        Combined DataFrame with all metrics
    """
    # Merge the DataFrames
    combined_df = pd.merge(
        insights_df,
        # Excluding campaign_name from status as it's already in insights
        status_df[['campaign_id', 'delivery_status',
                   'ad_start_time', 'ad_stop_time']],
        on='campaign_id',
        how='left'  # Keep all rows from insights_df
    )

    # Reorder columns for better readability
    column_order = [
        'date',
        'campaign_id',
        'campaign_name',
        'delivery_status',
        'spend',
        'reach',
        'link_click',
        'frequency',
        'impressions',
        'results',
        'result_type',
        'cost_per_result',
        'delivery_level',
        'file_path',
        'date_start',
        'date_stop',
        'report_start_date',
        'report_end_date',
        'updated_at',
        'ad_start_time',
        'ad_stop_time'
    ]

    combined_df = combined_df[column_order]

    return combined_df


def combined_fb_data_uploader(db_credentials, combined_df, table_name='raw_spend_facebook'):
    """
    Upload combined Facebook data to Postgres database.
    Deletes existing data for the date range (last 3 days: T-3 to T-1) before inserting new data.
    """
    try:
        # Convert date columns to datetime
        date_columns = ['date', 'end_date', 'start_date', 'report_start_date',
                        'report_end_date', 'updated_at', 'ad_start_time', 'ad_stop_time']

        for col in date_columns:
            if col in combined_df.columns:
                combined_df[col] = pd.to_datetime(
                    combined_df[col], errors='coerce')

        # Extract date range from combined_df
        start_date = combined_df['date'].min()
        end_date = combined_df['date'].max()

        # Connect to the Postgres database
        conn = psycopg2.connect(
            host=db_credentials['hostname'],
            database=db_credentials['database'],
            user=db_credentials['username'],
            password=db_credentials['password'],
            port=db_credentials['port']
        )
        cursor = conn.cursor()

        # Set the schema
        schema_query = f"SET search_path TO {db_credentials['schema']};"
        cursor.execute(schema_query)

        # Delete existing data for the date range
        delete_query = f"""
            DELETE FROM {table_name}
            WHERE date >= %s AND date <= %s;
        """
        cursor.execute(delete_query, (start_date, end_date))

        # Prepare the data for insertion
        columns_mapping = {
            'date': 'date',
            'campaign_id': 'campaign_id',
            'campaign_name': 'ad_name',
            'delivery_status': 'delivery_status',
            'spend': 'spend_hkd',
            'reach': 'reach',
            'link_click': 'link_clicks',
            'frequency': 'frequency',
            'impressions': 'impression',
            'results': 'results',
            'result_type': 'result_type',
            'cost_per_result': 'cost_per_result',
            'delivery_level': 'delivery_level',
            'file_path': 'File Paths',
            'date_start': 'start_date',
            'date_stop': 'end_date',
            'report_start_date': 'report_start_date',
            'report_end_date': 'report_end_date',
            'updated_at': 'updated_at',
            'ad_start_time': 'Ad start time',
            'ad_stop_time': 'Ad stop time'
        }

        # Rename columns to match database schema
        df_to_upload = combined_df.rename(columns=columns_mapping)

        # Insert data in chunks
        chunk_size = 50
        chunks = [df_to_upload[i:i + chunk_size]
                  for i in range(0, len(df_to_upload), chunk_size)]

        for chunk in chunks:
            values_list = []
            for _, row in chunk.iterrows():
                values = []
                for value in row.values:
                    if pd.isnull(value):
                        values.append("NULL")
                    elif isinstance(value, str):
                        value = value.replace("'", "''")
                        values.append(f"'{value}'::text")
                    elif isinstance(value, (datetime, pd.Timestamp)):
                        values.append(f"'{value}'::timestamp")
                    elif isinstance(value, (int, float)):
                        values.append(str(value))
                    else:
                        values.append(f"'{value}'::text")
                values_list.append(f"({','.join(values)})")

            column_names = ', '.join(
                [f'"{col}"' for col in df_to_upload.columns])
            insert_query = f"""
                INSERT INTO {table_name} ({column_names}) 
                VALUES {','.join(values_list)}
            """
            cursor.execute(insert_query)
            conn.commit()
            print(f"Uploaded chunk of {len(chunk)} records")

        print(
            f"Successfully uploaded data for date range: {start_date} to {end_date}")

    except Exception as e:
        print(f"Error uploading combined data to table {table_name}: {e}")
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()

# ======================
# Start of Procedure Methods
# ======================


def calculate_dates(days_back=3, timezone='Asia/Hong_Kong'):
    """Calculate date range for the pipeline."""
    hkt = pytz.timezone(timezone)
    today = datetime.now(hkt).date()
    start_date = (today - timedelta(days=days_back)).strftime('%Y-%m-%d')
    end_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')
    return start_date, end_date


def get_credentials_with_validation():
    """Get and validate credentials."""
    fb_credentials, db_credentials = get_credentials()
    if not fb_credentials or not db_credentials:
        raise ValueError("Missing credentials in config file")
    return fb_credentials, db_credentials


def fetch_and_process_insights(credentials, start_date, end_date, level='campaign'):
    """Handle insights data pipeline."""
    print("Fetching campaign insight data...")
    fb_insights_result = fb_insights_caller(
        credentials[0], start_date, end_date, level)

    print("Processing FB insights...")
    insights_df = process_fb_insights(fb_insights_result)

    if insights_df.empty:
        raise ValueError("No insights data retrieved")
    return insights_df


def fetch_and_process_status(credentials, start_date, end_date, level='campaign'):
    """Handle status data pipeline."""
    print("Fetching campaign status data...")
    fb_status_result = fb_status_caller(
        credentials[0], start_date, end_date, level)

    print("Processing FB status...")
    status_df = process_fb_status(fb_status_result)

    if status_df.empty:
        raise ValueError("No status data retrieved")
    return status_df


def combine_datasets(insights_df, status_df):
    """Merge and validate combined data."""
    print("Combining FB data...")
    combined_df = combine_fb_data(insights_df, status_df)

    if combined_df.empty:
        raise ValueError("Combined dataframe is empty")
    return combined_df


def upload_to_database(credentials, dataframe, table_name='raw_spend_facebook'):
    """Handle database upload."""
    print("Uploading to database...")
    combined_fb_data_uploader(credentials[1], dataframe, table_name)
    print("FB Data has been uploaded to DB.")

# ======================
# END of Procedure Methods
# ======================


def main():
    """Orchestrates the entire pipeline."""
    try:
        # Initialize
        start_date, end_date = calculate_dates()
        credentials = get_credentials_with_validation()

        # Execute pipeline
        insights_data = fetch_and_process_insights(
            credentials, start_date, end_date)
        status_data = fetch_and_process_status(
            credentials, start_date, end_date)
        combined_data = combine_datasets(insights_data, status_data)
        upload_to_database(credentials, combined_data)

        print("Pipeline completed successfully!")
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Application failed: {str(e)}")
