import pandas as pd
import numpy as np


df = pd.read_parquet('../data/us_accidents.parquet')

df.drop(columns=['ID', 'Source', 'Distance(mi)', 'End_Time', 'Duration', 'End_Lat', 'End_Lng', 'Description','Country', 'Street', 'Weather_Timestamp', 'Timezone', 'Zipcode'], inplace = True , errors='ignore')

df_airports = pd.read_csv('../data/data-world-us-airports.csv')

df_airports.drop(columns=['type', 'elevation_ft', 'continent', 'iso_country', 'iso_region', 'municipality', 'icao_code', 'iata_code'], inplace=True, errors='ignore')
df_airports[['latitude', 'longitude']] = df_airports['coordinates'].str.split(',', expand=True)
df_airports['latitude'] = df_airports['latitude'].astype(float)
df_airports['longitude'] = df_airports['longitude'].astype(float)
df_airports.drop(columns = ['coordinates'], inplace=True, errors='ignore')
df_airports.rename(columns={'ident': 'Airport_Code', 'name':'airport_name'}, inplace=True, errors='ignore')

df = df.merge(df_airports, how='left', left_on='Airport_Code', right_on='Airport_Code', suffixes=('', '_airport'))


airport_corrections = {
    'KCQT': {
        'airport_name': 'Whiteman Airport',
        'latitude': 34.2598,
        'longitude': -118.4119
    },
    'KMCJ': {
        'airport_name': 'William P Hobby Airport',
        'latitude': 29.6459,
        'longitude': -95.2769
    },
    'KATT': {
        'airport_name': 'Austin–Bergstrom International Airport',
        'latitude': 30.197535,
        'longitude': -97.662015
    },
    'KJRB': {
        'airport_name': 'Downtown Manhattan Heliport',
        'latitude': 40.700711,
        'longitude': -74.008573
    },
    'KDMH': {
        'airport_name': 'Baltimore/Washington International Airport',
        'latitude': 39.1718,
        'longitude': -76.6677
    }

}

for code, info in airport_corrections.items():
    mask = df['Airport_Code'] == code

    df.loc[mask, 'airport_name'] = info['airport_name']
    df.loc[mask, 'latitude'] = info['latitude']
    df.loc[mask, 'longitude'] = info['longitude']



conditional_airport_data = {
    'KNYC': {
        'Kings': {
            'airport_name': 'John F. Kennedy International Airport (JFK)',
            'latitude': 40.64013,
            'longitude': -73.77651
        },
        'New York': {
            'airport_name': 'LaGuardia Airport (LGA)',
            'latitude': 40.7769,
            'longitude': -73.8740
        },
        'Hudson': {
            'airport_name': 'Newark Liberty International Airport (EWR)',
            'latitude': 40.6925,
            'longitude': -74.1687
        },
        'Queens': {
            'airport_name': 'John F. Kennedy International Airport (JFK)',
            'latitude': 40.64013,
            'longitude': -73.77651
        },
        'Bergen': {
            'airport_name': 'Teterboro Airport (TEB)',
            'latitude': 40.8502,
            'longitude': -74.0609
        }
    }
}


for code, county_map in conditional_airport_data.items():
    for county, info in county_map.items():
        mask = (df['Airport_Code'] == code) & (df['County'] == county)

        for col, value in info.items():
            df.loc[mask, col] = value

df.dropna(subset=['Airport_Code'], inplace=True)

df['airport_name'].fillna('Unknown', inplace=True)

weather_cols = [
    'Temperature(F)','Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 
    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)'
]

for col in weather_cols:
    df[col] = df.groupby('Airport_Code')[col].transform(lambda x: x.fillna(x.median()))

for col in weather_cols:
    df[col].fillna(df[col].median(), inplace=True)


df.drop(columns=['Airport_Code', 'gps_code', 'local_code'], inplace=True, errors='ignore')
df.rename(columns={'latitude': 'airport_latitude', 'longitude': 'airport_longitude'}, inplace=True, errors='ignore')


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the earth (specified in decimal degrees).
    """
    # Check for missing values
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan

    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in miles. Use 6371 for kilometers
    r = 3956
    return c * r

df['distance_to_airport(mi)'] = df.apply(
    lambda row: haversine_distance(
        row['Start_Lat'], 
        row['Start_Lng'], 
        row['airport_latitude'], 
        row['airport_longitude']
    ), 
    axis=1
)

df.drop(columns=['Start_Lat', 'Start_Lng', 'City', 'County','Airport_Code', 'airport_name', 'airport_latitude', 'airport_longitude', 'Wind_Chill(F)'], inplace=True, errors='ignore')

df.to_parquet('../data/processed/us_accidents_pandas.parquet', index=False, engine='pyarrow')