#!/usr/bin/env python3
"""
Refactored dataPrepPandas.py
- Resolves FileNotFoundError by resolving default paths relative to this script.
- Adds argparse, logging, file existence checks.
- Uses vectorized operations (no row-wise apply) for distance calculation.
- Keeps processing logic equivalent to original but safer and modular.
"""
from pathlib import Path
import argparse
import logging
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def resolve_default(path: Path) -> Path:
    return path.expanduser().resolve()


def load_accidents(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        logger.error("Accidents file not found: %s", parquet_path)
        raise FileNotFoundError(parquet_path)
    logger.info("Loading accidents from %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    drop_cols = [
        "ID", "Source", "Distance(mi)", "End_Time", "Duration",
        "End_Lat", "End_Lng", "Description", "Country", "Street",
        "Weather_Timestamp", "Timezone", "Zipcode"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    logger.info("Accidents loaded: rows=%d cols=%d", df.shape[0], df.shape[1])
    return df


def load_airports(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        logger.error("Airports file not found: %s", csv_path)
        raise FileNotFoundError(csv_path)
    logger.info("Loading airports from %s", csv_path)
    df_air = pd.read_csv(csv_path, dtype=str)
    drop_cols = ['type', 'elevation_ft', 'continent', 'iso_country', 'iso_region',
                 'municipality', 'icao_code', 'iata_code']
    df_air = df_air.drop(columns=[c for c in drop_cols if c in df_air.columns], errors="ignore")

    # Robust extract of coordinates: latitude, longitude (first two numeric tokens)
    if "coordinates" in df_air.columns:
        coords = df_air["coordinates"].astype(str).str.extract(r"\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)")
        coords.columns = ["latitude", "longitude"]
        df_air = df_air.join(coords)
        # cast to float where possible
        df_air["latitude"] = pd.to_numeric(df_air["latitude"], errors="coerce")
        df_air["longitude"] = pd.to_numeric(df_air["longitude"], errors="coerce")
        df_air = df_air.drop(columns=["coordinates"], errors="ignore")

    df_air = df_air.rename(columns={"ident": "Airport_Code", "name": "airport_name"})
    logger.info("Airports loaded: rows=%d cols=%d", df_air.shape[0], df_air.shape[1])
    return df_air


def apply_airport_corrections(df: pd.DataFrame) -> pd.DataFrame:
    airport_corrections = {
        'KCQT': {'airport_name': 'Whiteman Airport', 'latitude': 34.2598, 'longitude': -118.4119},
        'KMCJ': {'airport_name': 'William P Hobby Airport', 'latitude': 29.6459, 'longitude': -95.2769},
        'KATT': {'airport_name': 'Austin–Bergstrom International Airport', 'latitude': 30.197535, 'longitude': -97.662015},
        'KJRB': {'airport_name': 'Downtown Manhattan Heliport', 'latitude': 40.700711, 'longitude': -74.008573},
        'KDMH': {'airport_name': 'Baltimore/Washington International Airport', 'latitude': 39.1718, 'longitude': -76.6677}
    }
    for code, info in airport_corrections.items():
        mask = df["Airport_Code"] == code
        if mask.any():
            df.loc[mask, "airport_name"] = info["airport_name"]
            df.loc[mask, "latitude"] = info["latitude"]
            df.loc[mask, "longitude"] = info["longitude"]
    return df


def apply_conditional_airport_data(df: pd.DataFrame) -> pd.DataFrame:
    conditional_airport_data = {
        'KNYC': {
            'Kings': {'airport_name': 'John F. Kennedy International Airport (JFK)', 'latitude': 40.64013, 'longitude': -73.77651},
            'New York': {'airport_name': 'LaGuardia Airport (LGA)', 'latitude': 40.7769, 'longitude': -73.8740},
            'Hudson': {'airport_name': 'Newark Liberty International Airport (EWR)', 'latitude': 40.6925, 'longitude': -74.1687},
            'Queens': {'airport_name': 'John F. Kennedy International Airport (JFK)', 'latitude': 40.64013, 'longitude': -73.77651},
            'Bergen': {'airport_name': 'Teterboro Airport (TEB)', 'latitude': 40.8502, 'longitude': -74.0609}
        }
    }
    for code, county_map in conditional_airport_data.items():
        for county, info in county_map.items():
            mask = (df["Airport_Code"] == code) & (df.get("County") == county)
            if mask.any():
                for col, val in info.items():
                    df.loc[mask, col] = val
    return df


def fill_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    weather_cols = [
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)'
    ]
    existing = [c for c in weather_cols if c in df.columns]
    if not existing:
        return df

    # group-wise median per Airport_Code (fallbacks handled by global median)
    group_medians = df.groupby("Airport_Code")[existing].median().rename_axis("Airport_Code").reset_index()
    # merge medians to accidents df to use for filling
    df = df.merge(group_medians, how="left", on="Airport_Code", suffixes=("", "_grpmed"))

    for col in existing:
        grpmed_col = f"{col}_grpmed"
        global_med = df[col].median()
        # fill: use group median where available, else global median
        df[col] = df[col].fillna(df[grpmed_col]).fillna(global_med)
        # drop helper
        df = df.drop(columns=[grpmed_col], errors="ignore")

    return df


def haversine_vectorized(lat1, lon1, lat2, lon2):
    # Inputs are pandas Series or array-like; returns numpy array
    mask = (lat1.notna()) & (lon1.notna()) & (lat2.notna()) & (lon2.notna())
    out = np.full(len(lat1), np.nan, dtype=float)
    if not mask.any():
        return out

    lat1_r = np.radians(lat1[mask].astype(float).to_numpy())
    lon1_r = np.radians(lon1[mask].astype(float).to_numpy())
    lat2_r = np.radians(lat2[mask].astype(float).to_numpy())
    lon2_r = np.radians(lon2[mask].astype(float).to_numpy())

    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3956.0  # miles
    out_vals = c * r
    out[mask.to_numpy()] = out_vals
    return out


def calculate_distance(df: pd.DataFrame) -> pd.DataFrame:
    # ensure airport coords are named airport_latitude/airport_longitude
    if "latitude" in df.columns and "longitude" in df.columns:
        df = df.rename(columns={"latitude": "airport_latitude", "longitude": "airport_longitude"})
    lat1 = df.get("Start_Lat")
    lon1 = df.get("Start_Lng")
    lat2 = df.get("airport_latitude")
    lon2 = df.get("airport_longitude")
    df["distance_to_airport(mi)"] = haversine_vectorized(lat1, lon1, lat2, lon2)
    return df


def final_cleanup_and_write(df: pd.DataFrame, out_path: Path):
    final_drop = [
        "Start_Lat", "Start_Lng", "City", "County", "Airport_Code",
        "airport_name", "airport_latitude", "airport_longitude", "Wind_Chill(F)", "gps_code", "local_code"
    ]
    df = df.drop(columns=[c for c in final_drop if c in df.columns], errors="ignore")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing processed data to %s", out_path)
    df.to_parquet(out_path, index=False, engine="pyarrow")


def run(input_parquet: Path, airports_csv: Path, output_parquet: Path):
    df = load_accidents(input_parquet)
    df_air = load_airports(airports_csv)
    # left join like original
    df = df.merge(df_air, how="left", left_on="Airport_Code", right_on="Airport_Code", suffixes=("", "_airport"))

    df = apply_airport_corrections(df)
    df = apply_conditional_airport_data(df)

    # drop rows missing Airport_Code if that behavior is desired (keeps as original)
    if "Airport_Code" in df.columns:
        before = len(df)
        df = df.dropna(subset=["Airport_Code"])
        logger.info("Dropped %d rows without Airport_Code", before - len(df))

    # fill unknown names
    if "airport_name" in df.columns:
        df["airport_name"] = df["airport_name"].fillna("Unknown")

    df = fill_weather_data(df)
    # rename coords if present
    if "latitude" in df.columns and "longitude" in df.columns:
        df = df.rename(columns={"latitude": "airport_latitude", "longitude": "airport_longitude"})
    df = calculate_distance(df)
    final_cleanup_and_write(df, output_parquet)
    logger.info("Processing complete. Output: %s", output_parquet)


def parse_args(argv):
    p = argparse.ArgumentParser(description="Process US accidents data (pandas)")
    default_input = resolve_default(DEFAULT_DATA_DIR / "us_accidents.parquet")
    default_airports = resolve_default(DEFAULT_DATA_DIR / "data-world-us-airports.csv")
    default_output = resolve_default(DEFAULT_DATA_DIR / "processed" / "us_accidents_pandas.parquet")
    p.add_argument("--input", "-i", type=Path, default=default_input, help=f"Input parquet (default: {default_input})")
    p.add_argument("--airports", "-a", type=Path, default=default_airports, help=f"Airports csv (default: {default_airports})")
    p.add_argument("--output", "-o", type=Path, default=default_output, help=f"Output parquet (default: {default_output})")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args.input, args.airports, args.output)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(2)
    except Exception as e:
        logger.exception("Unhandled error during processing: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()