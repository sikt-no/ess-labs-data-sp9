from pathlib import Path

import geopandas
import httpx
import pandas as pd
from rich.progress import track
import rioxarray as rxr
from shapely.geometry import LineString

from config import EEA_PATH, ERA5_PATH, TMP_PATH, GLOBAL_POP_FILE, NUTS_PATH

REGIONS = [
    "AT13",
    "BE10",
    "CZ010",
    "DE3",
    "ES30",
    "FR10",
    "HU101",
    "HU110",
    "NO01",
    "SE11",
    "SE110",
    "UKI",
]
TIMEZONES = {
    "AT13": "Europe/Vienna",
    "BE10": "Europe/Brussels",
    "CZ010": "Europe/Prague",
    "DE3": "Europe/Berlin",
    "ES30": "Europe/Madrid",
    "FR10": "Europe/Paris",
    "HU101": "Europe/Budapest",
    "HU110": "Europe/Budapest",
    "NO01": "Europe/Oslo",
    "SE110": "Europe/Stockholm",
    "SE11": "Europe/Stockholm",
    "UKI": "Europe/London",
}

ESS_FILES = [
    "ESS8e02_2.sav",
    "ESS9e03_1.sav",
    "ESS10.sav",
    "ESS10SC.sav",
]


for path in [EEA_PATH, ERA5_PATH, TMP_PATH, NUTS_PATH]:
    if not path.exists():
        path.mkdir()


async def a_maybe_download(url, folder=None):
    path = make_path(url, folder)
    if not path.exists():
        async with httpx.AsyncClient() as client:
            res = await client.get(url, timeout=None)
        res.raise_for_status()
        data = res.content
        with open(path, "wb") as f:
            f.write(data)
    return path


def maybe_download(url, folder=None):
    path = make_path(url, folder)
    if not path.exists():
        res = httpx.get(url, timeout=None)
        res.raise_for_status()
        data = res.content
        with open(path, "wb") as f:
            f.write(data)
    return path


def make_path(url, folder):
    filename = url.split("/")[-1]
    path = Path(filename)
    if folder is not None:
        Path(folder).mkdir(exist_ok=True)
        path = Path(folder) / path
    return path


def load_nuts(folder=NUTS_PATH):
    def make_level_df(level, year):
        url = f"https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_{year}_4326_LEVL_{level}.geojson"
        path = maybe_download(url, folder)

        cols = ["NUTS_ID", "LEVL_CODE", "CNTR_CODE", "geometry"]
        level_df = geopandas.read_file(path).set_index("id")[cols]
        return level_df

    print("load_nuts")
    nuts_df = pd.DataFrame()
    for i in track(range(1, 4), description="Loading nuts"):
        level_df = make_level_df(i, 2016)
        nuts_df = pd.concat([nuts_df, level_df])

    # Missing for HU
    level_df = make_level_df(3, 2013)
    level_df = level_df[level_df["NUTS_ID"] == "HU101"]
    nuts_df = pd.concat([nuts_df, level_df])

    nuts_df = nuts_df.sort_index()
    print("NUTS_DF", nuts_df)
    return nuts_df


def load_geostat(path: str, region_df: geopandas.GeoDataFrame) -> pd.DataFrame:
    print("load_geostat")
    region_df = region_df.to_crs("ESRI:54009")
    dataarray = rxr.open_rasterio(path)
    geometry = region_df.iloc[0].geometry
    dfs = []
    for i in track(dataarray[0], description=f"loading geostat data"):
        i = i[i != dataarray.attrs["_FillValue"]]
        if len(i) == 0:
            continue
        s = i.to_pandas()
        y_val = int(i.coords.get("y").item())
        line = LineString([(s.index[0], y_val), (s.index[-1], y_val)])
        if geometry.crosses(line):
            df = s.reset_index()
            df["y"] = y_val
            dfs.append(df)
    conc = pd.concat(dfs, ignore_index=True)
    rdf = geopandas.GeoDataFrame(
        conc, geometry=geopandas.points_from_xy(conc.x, conc.y), crs="ESRI:54009"
    )
    pop_region_df = region_df.overlay(rdf, keep_geom_type=False)
    return pop_region_df


def filter_none(container):
    return [item for item in container if item is not None]


EEA_VARIABLE_LABELS = {
    "date": "Date",
    "aqiwdpm10": "Worst air quality index level PM10, date",
    "aqiwdpm2_5": "Worst air quality index level PM2.5, date",
    "aqiwdso2": "Worst air quality index level SO2, date",
    "aqiwdno2": "Worst air quality index level NO2, date",
    "aqiwdo3": "Worst air quality index level O3, date",
    "aqiwd": "Worst air quality index level across pollutants, date",
    "aqiw2dpm10": "Worst air quality index level PM10, last two days",
    "aqiw2dpm2_5": "Worst air quality index level PM2.5,  last two days",
    "aqiw2dso2": "Worst air quality index level SO2, last two days",
    "aqiw2dno2": "Worst air quality index level NO2, last two days",
    "aqiw2do3": "Worst air quality index level O3, last two days",
    "aqiw2d": "Worst air quality index level across pollutants, last two days",
    "ndyprwpm10": "Number of days with 'poor' air quality level or worse on PM10, week before the date",
    "ndyprwpm2_5": "Number of days with 'poor' air quality level or worse on PM2.5, week before the date",
    "ndyprwso2": "Number of days with 'poor' air quality level or worse on SO2, week before the date",
    "ndyprwno2": "Number of days with 'poor' air quality level or worse on NO2, week before the date",
    "ndyprwo3": "Number of days with 'poor' air quality level or worse on O3, week before the date",
    "ndyprw": "Number of days with  'Poor' air quality level or worse on one or more pollutant indicators, week before the date",
    "ndyprmpm10": "Number of days with 'poor' air quality level or worse on PM10, month before the date",
    "ndyprmpm2_5": "Number of days with 'poor' air quality level or worse PM2.5, month before the date",
    "ndyprmso2": "Number of days with 'poor' air quality level or worse SO2, month before the date",
    "ndyprmno2": "Number of days with 'poor' air quality level or worse NO2, month before the date",
    "ndyprmo3": "Number of days with 'poor' air quality level or worse O3, month before the date",
    "ndyprm": "Number of days with days with 'poor' level or worse on one or more pollutant indicators, month before the date",
    "ndyprypm10": "Number of days with 'poor' air quality level or worse on PM10, year before the date",
    "ndyprypm2_5": "Number of days with 'poor' air quality level or worse on PM2.5, year before the date",
    "ndypryso2": "Number of days with 'poor' air quality level or worse on SO2, year before the date",
    "ndypryno2": "Number of days with 'poor' air quality level or worse on NO2, year before the date",
    "ndypryo3": "Number of days with 'poor' air quality level or worse on O3, year before the date",
    "ndypry": "Number of days with days with 'poor' level or worse on one or more pollutant indicators, year before the date",
}
EEA_VALUE_LABELS = {
    0: "Good",
    1: "Fair",
    2: "Moderate",
    3: "Poor",
    4: "Very Poor",
    5: "Extremely poor",
}
EEA_VARIABLE_VALUE_LABELS = {
    var: EEA_VALUE_LABELS
    for var in [
        "aqiwdpm10",
        "aqiwdpm2_5",
        "aqiwdso2",
        "aqiwdno2",
        "aqiwdo3",
        "aqiwd",
        "aqiw2dpm10",
        "aqiw2dpm2_5",
        "aqiw2dso2",
        "aqiw2dno2",
        "aqiw2do3",
        "aqiw2d",
    ]
}
EEA_VARIABLE_FORMATS = {
    "aqiwdpm10": "F2.0",
    "aqiwdpm2_5": "F2.0",
    "aqiwdso2": "F2.0",
    "aqiwdno2": "F2.0",
    "aqiwdo3": "F2.0",
    "aqiwd": "F2.0",
    "aqiw2dpm10": "F2.0",
    "aqiw2dpm2_5": "F2.0",
    "aqiw2dso2": "F2.0",
    "aqiw2dno2": "F2.0",
    "aqiw2do3": "F2.0",
    "aqiw2d": "F2.0",
    "ndyprwpm10": "F4.0",
    "ndyprwpm2_5": "F4.0",
    "ndyprwso2": "F4.0",
    "ndyprwno2": "F4.0",
    "ndyprwo3": "F4.0",
    "ndyprw": "F4.0",
    "ndyprmpm10": "F4.0",
    "ndyprmpm2_5": "F4.0",
    "ndyprmso2": "F4.0",
    "ndyprmno2": "F4.0",
    "ndyprmo3": "F4.0",
    "ndyprm": "F4.0",
    "ndyprypm10": "F4.0",
    "ndyprypm2_5": "F4.0",
    "ndypryso2": "F4.0",
    "ndypryno2": "F4.0",
    "ndypryo3": "F4.0",
    "ndypry": "F4.0",
}
ERA5_VARIABLE_VALUES = {
    "date": "Date",
    "region": "Region",
    "tmpdca": "Temperature in degrees Celcius, date average",
    "tmpdcmx": "Temperature in degrees Celcius, date maximum",
    "tmpdcmn": "Temperature in degrees Celcius, date minmum",
    "tmpdcaw": "Temperature in degrees Celcius, week average before the date",
    "tmpdcam": "Temperature in degrees Celcius, month average before the date",
    "tmpdca3m": "Temperature in degrees Celcius, three months average before the date",
    "tmpdcay": "Temperature in degrees Celcius, year average before the date",
    "tmpdcacm": "Temperature in degrees Celcius, calendar month average",
    "tmpdcamb": "Temperature average in degrees Celcius, calendar month, baseline 1991 - 2020",
    "tmp95pacmb": "Temperature average of 95 percentil in degrees Celcius, calendar month, baseline 1991 - 2020",
    "tmpanod": "Temperature anomaly date",
    "tmpanocm": "Temperature anomaly calendar month",
    "paccta": "Total precipitation average, date",
    "pacctaw": "Total precipitation average, week",
    "pacctam": "Total precipitation average, month",
    "paccta3m": "Total precipitation average, three months",
    "pacctay": "Total precipitation average, year",
    "pacctcm": "Total precipitation, calendar month",
    "pacctmb": "Total precipitation, calendar month, baseline 1991 - 2020",
    "paccdcm": "Total precipitation - calendar month, deviation from normal",
    "iwg10mx": "Instantaneous 10 metre wind gust maximum, date",
    "iwg10mxaw": "Instantaneous 10 metre wind gust average, week",
    "iwg10mxam": "Instantaneous 10 metre wind gust average, month",
    "iwg10mxa3m": "Instantaneous 10 metre wind gust average maximum for the region, three months",
    "iwg10mxay": "Instantaneous 10 metre wind gust average maximum for the region, year",
    "iwg10mxamb": "Instantaneous 10 metre wind gust average maximum, calendar month, baseline 1991 - 2020",
}
