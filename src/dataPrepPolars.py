#!/usr/bin/env python3
"""
US Accidents Data Processing - Polars Implementation
Processes US accidents data with airport information using Polars for high performance.
"""

import polars as pl
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def haversine_distance_vectorized(lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr) -> pl.Expr:
    """
    Vectorized haversine distance calculation for Polars.
    Calculate the great-circle distance between two points on earth (in decimal degrees).
    Returns distance in miles.
    """
    # Convert to radians
    lat1_rad = lat1 * pl.lit(np.pi / 180)
    lon1_rad = lon1 * pl.lit(np.pi / 180)
    lat2_rad = lat2 * pl.lit(np.pi / 180)
    lon2_rad = lon2 * pl.lit(np.pi / 180)
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = (dlat / 2).sin().pow(2) + lat1_rad.cos() * lat2_rad.cos() * (dlon / 2).sin().pow(2)
    c = 2 * a.sqrt().arcsin()
    
    # Earth radius in miles
    r = 3956
    return c * r


def load_and_clean_accidents_data(file_path: str) -> pl.DataFrame:
    """Load and perform initial cleaning of accidents data."""
    logger.info(f"Loading accidents data from {file_path}")
    
    # Columns to drop
    drop_cols = [
        'ID', 'Source', 'Distance(mi)', 'End_Time', 'Duration', 
        'End_Lat', 'End_Lng', 'Description', 'Country', 'Street', 
        'Weather_Timestamp', 'Timezone', 'Zipcode'
    ]
    
    df = pl.read_parquet(file_path)
    
    # Drop columns if they exist
    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    if existing_drop_cols:
        df = df.drop(existing_drop_cols)
    
    logger.info(f"Loaded accidents data with {df.height} rows and {df.width} columns")
    return df


def load_and_process_airports_data(file_path: str) -> pl.DataFrame:
    """Load and process airports data."""
    logger.info(f"Loading airports data from {file_path}")
    
    # Columns to drop
    drop_cols = [
        'type', 'elevation_ft', 'continent', 'iso_country', 
        'iso_region', 'municipality', 'icao_code', 'iata_code'
    ]
    
    df_airports = pl.read_csv(file_path)
    
    # Drop columns if they exist
    existing_drop_cols = [col for col in drop_cols if col in df_airports.columns]
    if existing_drop_cols:
        df_airports = df_airports.drop(existing_drop_cols)

    # Robustly parse "coordinates" into latitude/longitude without out-of-bounds errors.
    # Note: Keeps the same order as your pandas notebook: first value -> latitude, second -> longitude.
    df_airports = (
        df_airports
        .with_columns([
            pl.col("coordinates").str.extract(r"^\s*([-+]?\d*\.?\d+)").alias("lat_s"),
            pl.col("coordinates").str.extract(r",\s*([-+]?\d*\.?\d+)").alias("lon_s"),
        ])
        .with_columns([
            pl.col("lat_s").cast(pl.Float64, strict=False).alias("latitude"),
            pl.col("lon_s").cast(pl.Float64, strict=False).alias("longitude"),
        ])
        .drop(["coordinates", "lat_s", "lon_s"])
    )
    
    # Rename columns
    df_airports = df_airports.rename({
        'ident': 'Airport_Code',
        'name': 'airport_name'
    })
    
    logger.info(f"Processed airports data with {df_airports.height} rows")
    return df_airports


def apply_airport_corrections(df: pl.DataFrame) -> pl.DataFrame:
    """Apply manual corrections for specific airport codes."""
    logger.info("Applying airport corrections")
    
    airport_corrections = {
        'KCQT': {'airport_name': 'Whiteman Airport', 'latitude': 34.2598, 'longitude': -118.4119},
        'KMCJ': {'airport_name': 'William P Hobby Airport', 'latitude': 29.6459, 'longitude': -95.2769},
        'KATT': {'airport_name': 'Austin–Bergstrom International Airport', 'latitude': 30.197535, 'longitude': -97.662015},
        'KJRB': {'airport_name': 'Downtown Manhattan Heliport', 'latitude': 40.700711, 'longitude': -74.008573},
        'KDMH': {'airport_name': 'Baltimore/Washington International Airport', 'latitude': 39.1718, 'longitude': -76.6677}
    }
    
    # Build expressions for all corrections at once
    airport_name_expr = pl.col('airport_name')
    latitude_expr = pl.col('latitude')
    longitude_expr = pl.col('longitude')
    
    for code, info in airport_corrections.items():
        airport_name_expr = pl.when(pl.col('Airport_Code') == code).then(pl.lit(info['airport_name'])).otherwise(airport_name_expr)
        latitude_expr = pl.when(pl.col('Airport_Code') == code).then(pl.lit(info['latitude'])).otherwise(latitude_expr)
        longitude_expr = pl.when(pl.col('Airport_Code') == code).then(pl.lit(info['longitude'])).otherwise(longitude_expr)
    
    df = df.with_columns([
        airport_name_expr.alias('airport_name'),
        latitude_expr.alias('latitude'),
        longitude_expr.alias('longitude')
    ])
    
    return df


def apply_conditional_airport_data(df: pl.DataFrame) -> pl.DataFrame:
    """Apply conditional airport data based on county."""
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
    
    # Build conditional expressions
    airport_name_expr = pl.col('airport_name')
    latitude_expr = pl.col('latitude')
    longitude_expr = pl.col('longitude')
    
    for code, county_map in conditional_airport_data.items():
        for county, info in county_map.items():
            condition = (pl.col('Airport_Code') == code) & (pl.col('County') == county)
            
            airport_name_expr = pl.when(condition).then(pl.lit(info['airport_name'])).otherwise(airport_name_expr)
            latitude_expr = pl.when(condition).then(pl.lit(info['latitude'])).otherwise(latitude_expr)
            longitude_expr = pl.when(condition).then(pl.lit(info['longitude'])).otherwise(longitude_expr)
    
    df = df.with_columns([
        airport_name_expr.alias('airport_name'),
        latitude_expr.alias('latitude'),
        longitude_expr.alias('longitude')
    ])
    
    return df


def fill_weather_data(df: pl.DataFrame) -> pl.DataFrame:
    """Fill missing weather data using group-wise median and global median fallback."""
    logger.info("Filling missing weather data")
    
    weather_cols = [
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 
        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)'
    ]
    
    # Filter weather columns that exist in the dataframe
    existing_weather_cols = [col for col in weather_cols if col in df.columns]
    
    if not existing_weather_cols:
        logger.warning("No weather columns found in dataframe")
        return df
    
    # Calculate global medians first for fallback
    global_medians = {}
    for col in existing_weather_cols:
        global_medians[col] = df.select(pl.col(col).median()).item()
    
    # Fill with group-wise median, then global median
    for col in existing_weather_cols:
        df = df.with_columns([
            pl.col(col).fill_null(
                pl.col(col).median().over('Airport_Code')
            ).fill_null(global_medians[col]).alias(col)
        ])
    
    return df


def calculate_airport_distance(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate distance from accident location to airport."""
    logger.info("Calculating distances to airports")
    
    # Rename airport coordinates to avoid confusion
    df = df.rename({
        'latitude': 'airport_latitude',
        'longitude': 'airport_longitude'
    })
    
    # Calculate haversine distance using expression
    df = df.with_columns([
        haversine_distance_vectorized(
            pl.col('Start_Lat'),
            pl.col('Start_Lng'),
            pl.col('airport_latitude'),
            pl.col('airport_longitude')
        ).alias('distance_to_airport(mi)')
    ])
    
    return df


def final_cleanup(df: pl.DataFrame) -> pl.DataFrame:
    """Perform final cleanup and column removal."""
    logger.info("Performing final cleanup")
    
    # Columns to drop in final cleanup
    final_drop_cols = [
        'Start_Lat', 'Start_Lng', 'City', 'County', 'Airport_Code', 
        'airport_name', 'airport_latitude', 'airport_longitude', 
        'Wind_Chill(F)', 'gps_code', 'local_code'
    ]
    
    # Drop columns if they exist
    existing_drop_cols = [col for col in final_drop_cols if col in df.columns]
    if existing_drop_cols:
        df = df.drop(existing_drop_cols)
    
    return df


def main():
    """Main processing function."""
    try:
        # Define file paths
        accidents_path = 'data/us_accidents.parquet'
        airports_path = 'data/data-world-us-airports.csv'
        output_path = 'data/processed/us_accidents_polars.parquet'
        
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Load and process data
        df = load_and_clean_accidents_data(accidents_path)
        df_airports = load_and_process_airports_data(airports_path)
        
        # Merge dataframes
        logger.info("Merging accidents and airports data")
        df = df.join(df_airports, on='Airport_Code', how='left')
        
        # Apply corrections and processing
        df = apply_airport_corrections(df)
        df = apply_conditional_airport_data(df)
        
        # Remove rows without airport codes
        logger.info("Filtering out rows without Airport_Code")
        df = df.filter(pl.col('Airport_Code').is_not_null())
        
        # Fill missing airport names
        df = df.with_columns([
            pl.col('airport_name').fill_null('Unknown')
        ])
        
        # Fill weather data
        df = fill_weather_data(df)
        
        # Calculate airport distances
        df = calculate_airport_distance(df)
        
        # Final cleanup
        df = final_cleanup(df)
        
        # Save processed data
        logger.info(f"Saving processed data to {output_path}")
        df.write_parquet(output_path)
        
        logger.info(f"Processing complete! Final dataset has {df.height} rows and {df.width} columns")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()