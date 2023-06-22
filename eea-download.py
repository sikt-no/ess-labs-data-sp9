import asyncio
import urllib.parse

import geopandas
import httpx
import pandas as pd

from utils import EEA_PATH, REGIONS, a_maybe_download, filter_none, load_nuts

client = httpx.AsyncClient()

HOUR = pd.Timedelta(hours=1)
pollutants = {
    "5": "PM10",
    "1": "SO2",
    "7": "O3",
    "8": "NO2",
    "6001": "PM2_5",
}


async def load_stations():
    station_file = await a_maybe_download(
        "https://discomap.eea.europa.eu/map/fme/metadata/PanEuropean_metadata.csv"
    )
    all_stations = pd.read_csv(station_file, sep=r"\t", engine="python")
    all_stations = all_stations.groupby("AirQualityStation").first().reset_index()
    all_stations = all_stations[all_stations["AirQualityStationType"] == "background"]
    all_stations = geopandas.GeoDataFrame(
        all_stations,
        geometry=geopandas.points_from_xy(
            all_stations["Longitude"], all_stations["Latitude"]
        ),
        crs="EPSG:4326",
    )
    return all_stations


async def get_csvs_for_station_pollutant(station, pollutant, ccode):
    params = {
        "Pollutant": pollutant,
        "CountryCode": ccode,
        "Station": station,
        "Year_from": "1991",
        "Year_to": "2022",
        "Source": "All",
        "Output": "TEXT",
        "TimeCoverage": "Year",
    }
    query = urllib.parse.urlencode(params)
    list_url = (
        "https://fme.discomap.eea.europa.eu/fmedatastreaming/AirQualityDownload/AQData_Extract.fmw?"
        + query
    )
    print(list_url)
    r = await client.get(list_url, timeout=None)

    if r.status_code == 204:
        return
    urls = r.content.decode("utf-8-sig").splitlines()
    filepaths = [await a_maybe_download(url, folder=EEA_PATH / "raw") for url in urls]

    return filepaths


async def get_data_for_pollutant(pollutant, station, ccode, region_id):
    filepaths = await get_csvs_for_station_pollutant(station, pollutant, ccode)
    if filepaths is None:
        return
    dfs = [pd.read_csv(path) for path in filepaths if path is not None]
    station_pollutant_df = pd.concat(dfs)
    relevant_vars = [
        "DatetimeBegin",
        "DatetimeEnd",
        "AirQualityStation",
        "AirPollutant",
        "Concentration",
    ]
    station_pollutant_df = station_pollutant_df[relevant_vars]
    station_pollutant_df["pollutant"] = pollutant
    station_pollutant_df["region"] = region_id
    station_pollutant_df["station_id"] = station
    return station_pollutant_df


async def get_data_for_station(station, ccode, region_id):
    station_pollutant_dfs = await asyncio.gather(
        *[
            get_data_for_pollutant(pollutant, station, ccode, region_id)
            for pollutant in pollutants
        ]
    )
    station_pollutant_dfs = filter_none(station_pollutant_dfs)
    if not station_pollutant_dfs:
        return
    station_df = pd.concat(station_pollutant_dfs)
    return station_df


async def get_data_for_region(region_id, nuts_df, all_stations):
    print(region_id)
    ccode = region_id[:2].replace("UK", "GB")
    region_df = nuts_df[nuts_df["NUTS_ID"] == region_id]
    assert len(region_df) == 1, region_df
    region_geometry = region_df.iloc[0].geometry

    country_stations = all_stations[all_stations["Countrycode"] == ccode]
    region_stations = country_stations[country_stations.within(region_geometry)]
    print("region stations: ", len(region_stations))

    station_dfs = await asyncio.gather(
        *[
            get_data_for_station(station, ccode, region_id)
            for station in list(region_stations["AirQualityStation"])
        ]
    )
    station_dfs = filter_none(station_dfs)
    if not station_dfs:
        return
    region_df = pd.concat(station_dfs)
    return region_df


async def main():
    nuts_df = load_nuts()
    all_stations = await load_stations()
    region_dfs = await asyncio.gather(
        *[
            get_data_for_region(region_id, nuts_df, all_stations)
            for region_id in REGIONS
        ]
    )
    region_dfs = filter_none(region_dfs)
    df = pd.concat(region_dfs)
    print(df["region"].value_counts())
    df.to_parquet(EEA_PATH / "eea-stations.pqt")  # FIXME


if __name__ == "__main__":
    asyncio.run(main())
