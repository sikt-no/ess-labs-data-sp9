from itertools import product
from pathlib import Path

import cdsapi
import geopandas
import netCDF4
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rich.progress import track
import shapely

from utils import ERA5_PATH, GLOBAL_POP_FILE, REGIONS, load_geostat, load_nuts

MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# fmt: off
YEARS = [
    "1990", "1991", "1992", "1993", "1994", "1995", "1996", "1997", "1998", "1999", "2000", "2001", "2002", "2003", "2004", "2005",
    "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021",
    "2022"
]

DAYS = [
    '01', '02', '03',
    '04', '05', '06',
    '07', '08', '09',
    '10', '11', '12',
    '13', '14', '15',
    '16', '17', '18',
    '19', '20', '21',
    '22', '23', '24',
    '25', '26', '27',
    '28', '29', '30',
    '31',
]

TIMES = [
    '00:00', '01:00', '02:00',
    '03:00', '04:00', '05:00',
    '06:00', '07:00', '08:00',
    '09:00', '10:00', '11:00',
    '12:00', '13:00', '14:00',
    '15:00', '16:00', '17:00',
    '18:00', '19:00', '20:00',
    '21:00', '22:00', '23:00',
]
# fmt: on

STEP = 0.1
HALF_STEP = STEP / 2

era5_variables = {
    "tmpdc": "2m_temperature",
    "pac": "total_precipitation",
    "iwg10": "instantaneous_10m_wind_gust",
}
era5_cols = {
    "tmpdc": "t2m",
    "pac": "tp",
    "iwg10": "i10fg",
}


def download_era5(era5_id: str, region_id, region_geometry, year: str, month) -> Path:
    path = ERA5_PATH / "raw" / f"{era5_id}{region_id}y{year}m{month}.netcdf"
    if path.exists():
        return path
    c = cdsapi.Client()

    res = c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": era5_variables[era5_id],
            "grid": f"{STEP}/{STEP}",
            "area": region_geometry.bounds,  # N, W, S, E
            "year": year,
            "month": month,
            "day": DAYS,
            "time": TIMES,
        },
        str(path),
    )
    print(res)
    return path


def read_netcdf(era5_id, filepath: Path) -> pd.DataFrame:
    col = era5_cols[era5_id]
    ds = netCDF4.Dataset(str(filepath))
    A = ds[col]
    arranged = [ds[dim][:] for dim in A.dimensions]
    grids = np.meshgrid(*arranged)
    indices = np.stack([grid.flatten() for grid in grids], axis=1)
    df = pd.DataFrame(indices, columns=A.dimensions)
    df[era5_id] = np.asarray(A).flatten()
    df = df.rename(columns={"longitude": "latitude", "latitude": "longitude"})  # HM!
    df = df.set_index(["time", "longitude", "latitude"]).sort_index()
    df = df.astype("float32")
    return df


def make_month_df(region_id, region_geometry, year, month):
    month_measure_dfs = []
    for era5_id in era5_cols.keys():
        month_filepath = download_era5(
            era5_id, region_id, region_geometry, str(year), str(month)
        )
        month_measure_df = read_netcdf(era5_id, month_filepath)
        month_measure_dfs.append(month_measure_df)
    df = pd.concat(month_measure_dfs, axis=1).reset_index()
    df["time"] = df["time"].astype("int32")
    df["longitude"] = df["longitude"].astype("float32")
    df["latitude"] = df["latitude"].astype("float32")
    df["tmpdc"] = df["tmpdc"].astype("float32")
    df["pac"] = df["pac"].astype("float32")
    df["iwg10"] = df["iwg10"].astype("float32")
    return df


def merge_population_data(df_in, region_df) -> pd.Series:
    # grid_df: one row pr grid
    grid_df = df_in[["grid_id", "longitude", "latitude"]].drop_duplicates()
    grid_df["box"] = grid_df.apply(
        lambda x: shapely.geometry.Polygon(
            [
                [x["longitude"] - HALF_STEP, x["latitude"] - HALF_STEP],
                [x["longitude"] + HALF_STEP, x["latitude"] - HALF_STEP],
                [x["longitude"] + HALF_STEP, x["latitude"] + HALF_STEP],
                [x["longitude"] - HALF_STEP, x["latitude"] + HALF_STEP],
            ]
        ),
        axis=1,
    )
    grid_df = grid_df.drop(columns=["longitude", "latitude"])
    grid_df = geopandas.GeoDataFrame(grid_df, geometry="box", crs="EPSG:4326")

    # pop_df: one row per square meter in square containing region
    pop_df = load_geostat(GLOBAL_POP_FILE, region_df).rename(columns={0: "pop"})
    pop_df["LEVL_CODE"] = pop_df["LEVL_CODE"].astype("int8")
    pop_df["pop"] = pop_df["pop"].astype("float32")
    pop_df["x"] = pop_df["x"].astype("int32")
    pop_df["y"] = pop_df["y"].astype("int32")

    # grid_pop_df: one row per square meter in grid, with population in grid_id
    grid_pop_df = (
        grid_df.to_crs("EPSG:3035")
        .reset_index()
        .sjoin(pop_df.to_crs("EPSG:3035"), how="left", predicate="contains")
        .to_crs("EPSG:4326")
        .reset_index(drop=True)
    )

    pop_s = grid_pop_df.groupby("grid_id")["pop"].sum()
    return pop_s


def make_region_grids_table(nuts_df, region_id: str) -> pa.Table:
    path = f"tmp/tmp{region_id}era5.pqt"
    print(f"make_region_month_df {region_id}")
    region_df = nuts_df[nuts_df["NUTS_ID"] == region_id]
    assert len(region_df) == 1, region_df
    region_geometry = region_df.iloc[0].geometry

    month_dfs = [
        make_month_df(region_id, region_geometry, year, month)
        for year, month in track(list(product(YEARS, MONTHS)))
    ]
    df = pd.concat(month_dfs)
    df["region"] = region_id
    df["grid_id"] = df.groupby(["longitude", "latitude"]).ngroup()
    df["grid_id"] = df["grid_id"].astype("int16")

    population_by_grid = merge_population_data(df, region_df)
    df = df.merge(population_by_grid, left_on="grid_id", right_index=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)
    return path


def main():
    print("main")
    nuts_df = load_nuts()
    region_table_paths = [
        make_region_grids_table(nuts_df=nuts_df, region_id=region_id)
        for region_id in REGIONS
    ]
    region_tables = [pq.read_table(path) for path in region_table_paths]
    table = pa.concat_tables(region_tables)
    path = ERA5_PATH / "era5-grids.pqt"
    pq.write_table(table, path)


if __name__ == "__main__":
    main()
