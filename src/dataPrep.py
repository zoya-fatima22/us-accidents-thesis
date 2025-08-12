#!/usr/bin/env python3
"""
US Accidents Data Processing - Dask Implementation
Processes US accidents data with airport information using Dask for distributed computing.
"""

import dask.dataframe as dd
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from dask.distributed import Client
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the earth (specified in decimal degrees).
    Vectorized for use with Dask.
    """
    # Handle missing values
    mask = pd.isna(lat1) | pd.isna(lon1) | pd.isna(lat2) | pd.isna(lon2)
    
    # Convert decimal degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in miles
    r = 3956
    distance = c * r
    
    # Set missing values back to NaN
    distance[mask] = np.nan
    
    return distance


def load_and_clean_accidents_data(file_path: str, npartitions: int = None) -> dd.DataFrame:
    """Load and perform initial cleaning of accidents data."""
    logger.info(f"Loading accidents data from {file_path}")
    
    # Columns to drop
    drop_cols = [
        'ID', 'Source', 'Distance(mi)', 'End_Time', 'Duration', 
        'End_Lat', 'End_Lng', 'Description', 'Country', 'Street', 
        'Weather_Timestamp', 'Timezone', 'Zipcode'
    ]
    
    # Load data with Dask
    df = dd.read_parquet(file_path, engine='pyarrow')
    
    # Optimize partitions - too many partitions can cause overhead
    # Rule of thumb: aim for 100MB-1GB per partition
    if npartitions is None:
        # Calculate reasonable number of partitions based on data size
        memory_usage = df.memory_usage(deep=True).sum().compute()
        target_partition_size = 500 * 1024 * 1024  # 500MB per partition
        npartitions = max(1, int(memory_usage / target_partition_size))
        npartitions = min(npartitions, 100)  # Cap at 100 partitions for reasonable overhead
    
    df = df.repartition(npartitions=npartitions)
    
    # Drop columns if they exist
    existing_cols = df.columns.tolist()
    drop_cols_existing = [col for col in drop_cols if col in existing_cols]
    
    if drop_cols_existing:
        df = df.drop(columns=drop_cols_existing)
    
    logger.info(f"Loaded accidents data with {len(df)} partitions")
    return df


def load_and_process_airports_data(file_path: str) -> pd.DataFrame:
    """Load and process airports data. Keep as pandas for easier manipulation."""
    logger.info(f"Loading airports data from {file_path}")
    
    # Columns to drop
    drop_cols = [
        'type', 'elevation_ft', 'continent', 'iso_country', 
        'iso_region', 'municipality', 'icao_code', 'iata_code'
    ]
    
    df_airports = pd.read_csv(file_path)
    
    # Drop columns if they exist
    existing_cols = df_airports.columns.tolist()
    drop_cols_existing = [col for col in drop_cols if col in existing_cols]
    
    if drop_cols_existing:
        df_airports = df_airports.drop(columns=drop_cols_existing)
    
    # Split coordinates and convert to float
    coords_split = df_airports['coordinates'].str.split(',', expand=True)
    df_airports['latitude'] = pd.to_numeric(coords_split[0], errors='coerce')
    df_airports['longitude'] = pd.to_numeric(coords_split[1], errors='coerce')
    df_airports = df_airports.drop(columns=['coordinates'])
    
    # Rename columns
    df_airports = df_airports.rename(columns={
        'ident': 'Airport_Code',
        'name': 'airport_name'
    })
    
    logger.info(f"Processed airports data with {len(df_airports)} rows")
    return df_airports


def apply_airport_corrections(df_airports: pd.DataFrame) -> pd.DataFrame:
    """Apply manual corrections for specific airport codes."""
    logger.info("Applying airport corrections")
    
    airport_corrections = {
        'KCQT': {'airport_name': 'Whiteman Airport', 'latitude': 34.2598, 'longitude': -118.4119},
        'KMCJ': {'airport_name': 'William P Hobby Airport', 'latitude': 29.6459, 'longitude': -95.2769},
        'KATT': {'airport_name': 'Austin–Bergstrom International Airport', 'latitude': 30.197535, 'longitude': -97.662015},
        'KJRB': {'airport_name': 'Downtown Manhattan Heliport', 'latitude': 40.700711, 'longitude': -74.008573},
        'KDMH': {'airport_name': 'Baltimore/Washington International Airport', 'latitude': 39.1718, 'longitude': -76.6677}
    }
    
    for code, info in airport_corrections.items():
        mask = df_airports['Airport_Code'] == code
        for col, value in info.items():
            df_airports.loc[mask, col] = value
    
    return df_airports


def apply_conditional_airport_corrections(df: dd.DataFrame) -> dd.DataFrame:
    """Apply conditional airport data based on county using Dask operations."""
    logger.info("Applying conditional airport data")
    
    conditional_airport_data = {
        'KNYC': {
            'Kings': {'airport_name': 'John F. Kennedy International Airport (JFK)', 'latitude': 40.64013, 'longitude': -73.77651},
            'New York': {'airport_name': 'LaGuardia Airport (LGA)', 'latitude': 40.7769, 'longitude': -73.8740},
            'Hudson': {'airport_name': 'Newark Liberty International Airport (EWR)', 'latitude': 40.6925, 'longitude': -74.1687},
            'Queens': {'airport_name': 'John F. Kennedy International Airport (JFK)', 'latitude': 40.64013, 'longitude': -73.77651},
            'Bergen': {'airport_name': 'Teterboro Airport (TEB)', 'latitude': 40.8502, 'longitude': -74.0609}
        }
    }
    
    def apply_conditional_corrections(partition):
        for code, county_map in conditional_airport_data.items():
            for county, info in county_map.items():
                mask = (partition['Airport_Code'] == code) & (partition['County'] == county)
                for col, value in info.items():
                    partition.loc[mask, col] = value
        return partition
    
    df = df.map_partitions(apply_conditional_corrections, meta=df._meta)
    return df


def fill_weather_data_dask(df: dd.DataFrame) -> dd.DataFrame:
    """Fill missing weather data using Dask operations."""
    logger.info("Filling missing weather data")
    
    weather_cols = [
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 
        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)'
    ]
    
    # Filter weather columns that exist in the dataframe
    existing_cols = df.columns.tolist()
    existing_weather_cols = [col for col in weather_cols if col in existing_cols]
    
    if not existing_weather_cols:
        logger.warning("No weather columns found in dataframe")
        return df
    
    # Calculate global medians for fallback using approximate median
    logger.info("Calculating approximate global medians for weather data")
    global_medians = {}
    for col in existing_weather_cols:
        try:
            # Use approximate median which works with Dask
            global_medians[col] = df[col].quantile(0.5).compute()
        except Exception as e:
            logger.warning(f"Could not calculate median for {col}, using mean instead: {e}")
            global_medians[col] = df[col].mean().compute()
    
    # Pre-calculate airport-wise medians for more efficient group operations
    logger.info("Calculating airport-wise medians for weather data")
    airport_medians = {}
    for col in existing_weather_cols:
        try:
            # Group by airport and calculate quantile (median approximation)
            airport_median = df.groupby('Airport_Code')[col].quantile(0.5).compute()
            airport_medians[col] = airport_median.to_dict()
        except Exception as e:
            logger.warning(f"Could not calculate airport medians for {col}, using global median: {e}")
            airport_medians[col] = {}
    
    # Function to fill missing values within each partition
    def fill_weather_partition(partition):
        # Fill with airport-wise median first, then global median
        for col in existing_weather_cols:
            if col in partition.columns:
                # Create a copy to avoid SettingWithCopyWarning
                partition = partition.copy()
                
                # Fill with airport-wise median
                if col in airport_medians:
                    partition[col] = partition.apply(
                        lambda row: airport_medians[col].get(row['Airport_Code'], global_medians[col]) 
                        if pd.isna(row[col]) else row[col], axis=1
                    )
                
                # Fill remaining NaNs with global median
                partition[col] = partition[col].fillna(global_medians[col])
        
        return partition
    
    df = df.map_partitions(fill_weather_partition, meta=df._meta)
    return df


def calculate_airport_distance_dask(df: dd.DataFrame) -> dd.DataFrame:
    """Calculate distance from accident location to airport using Dask."""
    logger.info("Calculating distances to airports")
    
    # Rename airport coordinates to avoid confusion
    df = df.rename(columns={
        'latitude': 'airport_latitude',
        'longitude': 'airport_longitude'
    })
    
    # Calculate haversine distance using map_partitions
    def calculate_distance_partition(partition):
        partition['distance_to_airport(mi)'] = haversine_distance(
            partition['Start_Lat'].values,
            partition['Start_Lng'].values,
            partition['airport_latitude'].values,
            partition['airport_longitude'].values
        )
        return partition
    
    df = df.map_partitions(
        calculate_distance_partition,
        meta=df._meta.assign(**{'distance_to_airport(mi)': 'float64'})
    )
    
    return df


def final_cleanup_dask(df: dd.DataFrame) -> dd.DataFrame:
    """Perform final cleanup and column removal."""
    logger.info("Performing final cleanup")
    
    # Columns to drop in final cleanup
    final_drop_cols = [
        'Start_Lat', 'Start_Lng', 'City', 'County', 'Airport_Code', 
        'airport_name', 'airport_latitude', 'airport_longitude', 
        'Wind_Chill(F)', 'gps_code', 'local_code'
    ]
    
    # Drop columns if they exist
    existing_cols = df.columns.tolist()
    drop_cols_existing = [col for col in final_drop_cols if col in existing_cols]
    
    if drop_cols_existing:
        df = df.drop(columns=drop_cols_existing)
    
    return df


def main():
    """Main processing function."""
    # Initialize Dask client with better configuration
    client = Client(
        processes=False, 
        threads_per_worker=4, 
        n_workers=1, 
        memory_limit='4GB',
        silence_logs=False
    )
    logger.info(f"Dask client initialized: {client.dashboard_link}")
    
    try:
        # Define file paths
        accidents_path = '../data/us_accidents.parquet'
        airports_path = '../data/data-world-us-airports.csv'
        output_path = '../data/processed/us_accidents_cleaned_dask.parquet'
        
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Load and process data with reasonable partitioning
        logger.info("Loading data...")
        df = load_and_clean_accidents_data(accidents_path, npartitions=50)  # Reasonable partition count
        df_airports = load_and_process_airports_data(airports_path)
        
        # Apply corrections to airports data (pandas operations)
        df_airports = apply_airport_corrections(df_airports)
        
        # Merge dataframes
        logger.info("Merging accidents and airports data")
        df = df.merge(df_airports, on='Airport_Code', how='left', suffixes=('', '_airport'))
        
        # Apply conditional corrections (Dask operations)
        df = apply_conditional_airport_corrections(df)
        
        # Remove rows without airport codes
        logger.info("Filtering out rows without Airport_Code")
        df = df.dropna(subset=['Airport_Code'])
        
        # Fill missing airport names
        df = df.fillna({'airport_name': 'Unknown'})
        
        # Fill weather data
        df = fill_weather_data_dask(df)
        
        # Calculate airport distances
        df = calculate_airport_distance_dask(df)
        
        # Final cleanup
        df = final_cleanup_dask(df)
        
        # Save processed data
        logger.info(f"Saving processed data to {output_path}")
        df.to_parquet(output_path, engine='pyarrow', compression='snappy')
        
        # Get final dataset info (compute actual values)
        logger.info("Computing final dataset statistics...")
        nrows = len(df.compute())
        ncols = len(df.columns)
        logger.info(f"Processing complete! Final dataset has {nrows} rows and {ncols} columns")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()