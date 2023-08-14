from functools import partial

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyreadstat
import pytz
from rich.progress import track

from utils import ERA5_PATH, TMP_PATH, ERA5_VARIABLE_VALUES, REGIONS, TIMEZONES


def create_date_column(df):
    # make time variable timezone aware
    df["time"] = df["time"].dt.tz_localize(pytz.utc)
    df["date"] = pd.Series(dtype="object")
    for region_id, tz_name in TIMEZONES.items():
        df["date"] = df["date"].mask(
            cond=df["region"] == region_id,
            other=df["time"].dt.tz_convert(tz_name).dt.date,
        )
    df = df.drop(columns=["time"])
    return df


def fix_measurements(df):
    df["tmpdc"] = df["tmpdc"] - 273.15  # kelvin to celcius
    df["pac"] = (df["pac"] * 1000).round(2)  # meters to millimeters
    return df


def groupby_date(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate grid-based daily values
    """
    daily_grouper = df_in.groupby(["region", "grid_id", "date"])
    df = pd.DataFrame(
        {
            "pop": daily_grouper["pop"].first(),
            "tmpdca": daily_grouper["tmpdc"].mean(numeric_only=True),
            "tmpdcmx": daily_grouper["tmpdc"].max(),
            "tmpdcmn": daily_grouper["tmpdc"].min(),
            "paccta": daily_grouper["pac"].sum(),
            "iwg10mx": daily_grouper["iwg10"].max(),
        }
    )
    return df


def calculating_moving_averages(df):
    df = df.copy()
    df = df.reset_index()
    # convert 'date' from date dtype to datetime to facilitate rolling averages
    df = df.set_index(pd.to_datetime(df.pop("date"), utc=True))

    groupby = df.groupby("region")
    averages = pd.DataFrame(
        {
            "tmpdcaw": groupby["tmpdca"].rolling("7D").mean(),
            "tmpdcam": groupby["tmpdca"].rolling("30D").mean(),
            "tmpdca3m": groupby["tmpdca"].rolling("90D").mean(),
            "tmpdcay": groupby["tmpdca"].rolling("365D").mean(),
            "pacctaw": groupby["paccta"].rolling("7D").sum(),
            "pacctam": groupby["paccta"].rolling("30D").sum(),
            "paccta3m": groupby["paccta"].rolling("90D").sum(),
            "pacctay": groupby["paccta"].rolling("365D").sum(),
            "iwg10mxaw": groupby["iwg10mx"].rolling("7D").mean(),
            "iwg10mxam": groupby["iwg10mx"].rolling("30D").mean(),
            "iwg10mxa3m": groupby["iwg10mx"].rolling("90D").mean(),
            "iwg10mxay": groupby["iwg10mx"].rolling("365D").mean(),
        }
    )
    # convert 'date' from back to date
    averages = averages.reset_index()
    averages["date"] = averages["date"].dt.date
    averages = averages.set_index(["region", "date"])

    return averages


def calculating_anomalies(df_in):
    df = df_in.reset_index()
    df["cal_month"] = df["date"].apply(lambda val: val.month)
    df["year_month"] = df["date"].apply(lambda val: f"{val.year}-{val.month}")
    df_baseline = df[df["date"].apply(lambda val: 1991 <= val.year <= 2020)]
    tmp_cal_month_groupby = df_baseline.groupby("cal_month")["tmpdca"]

    tmpdcamb = tmp_cal_month_groupby.aggregate("mean")
    tmpdcamb.name = "tmpdcamb"
    df = df.join(tmpdcamb, on="cal_month")

    tmp95pacmb = tmp_cal_month_groupby.aggregate(partial(np.percentile, q=95))
    tmp95pacmb.name = "tmp95pacmb"
    df = df.join(tmp95pacmb, on="cal_month")

    df["tmpanod"] = df["tmpdca"] - df["tmpdcamb"]
    # more_than_95p = df["tmpdca"] > df["tmp95pacmb"]
    # df["tmp95p3d"] = more_than_95p.rolling(3, min_periods=3).sum()

    df["tmpdcacm"] = df.groupby("year_month")["tmpdca"].transform("mean")
    df["tmpanocm"] = df["tmpdcacm"] - df["tmpdcamb"]

    # paccta
    # sum daily to year_monthly first
    df["pacctcm"] = df.groupby("year_month")["paccta"].transform("sum")
    df_baseline["pacctcm"] = df_baseline.groupby("year_month")["paccta"].transform(
        "sum"
    )
    # calculate monthly for baseline, join back in
    pac_cal_month_groupby = df_baseline.groupby("cal_month")["pacctcm"]
    pacctmb = pac_cal_month_groupby.aggregate("mean")
    pacctmb.name = "pacctmb"
    df = df.join(pacctmb, on="cal_month")
    # anomalies
    df["paccdcm"] = (df["pacctcm"] / df["pacctmb"]) * 100

    iwg_cal_month_groupby = df_baseline.groupby("cal_month")["iwg10mx"]
    iwg10mxamb = iwg_cal_month_groupby.aggregate("mean")
    iwg10mxamb.name = "iwg10mxamb"
    df = df.join(iwg10mxamb, on="cal_month")

    df = df.set_index(["region", "date"])
    df = df[
        [
            "tmpdcamb",
            "tmp95pacmb",
            "tmpanod",
            # "tmp95p3d",
            "tmpdcacm",
            "tmpanocm",
            "pacctcm",
            "paccdcm",
            "pacctmb",
            "iwg10mxamb",
        ]
    ]
    return df


def weighted_average(group, var_name):
    return np.average(group[var_name], weights=group["pop"])


def collapse_grid(df: pd.DataFrame) -> pd.DataFrame:
    groupby = df.groupby(["region", "date"])
    cols = set(df.columns) - {"region", "grid_id", "date", "pop"}
    res = pd.DataFrame(
        {col: groupby.apply(partial(weighted_average, var_name=col)) for col in cols}
    )
    return res


def order_columns(df):
    var_order = [
        "date",
        "region",
        "tmpdca",
        "tmpdcmx",
        "tmpdcmn",
        "tmpdcaw",
        "tmpdcam",
        "tmpdca3m",
        "tmpdcay",
        "tmpdcacm",
        "tmpdcamb",
        "tmp95pacmb",
        "tmpanod",
        "tmpanocm",
        "paccta",
        "pacctaw",
        "pacctam",
        "paccta3m",
        "pacctay",
        "pacctcm",
        "pacctmb",
        "paccdcm",
        "iwg10mx",
        "iwg10mxam",
        "iwg10mxaw",
        "iwg10mxa3m",
        "iwg10mxay",
        "iwg10mxamb",
    ]
    # double check we are not adding or removing any columns
    diff = set(df.columns).symmetric_difference(var_order)
    assert not diff, diff
    df = df[var_order]
    return df


def do_for_region(region_id, path):

    hourly_grids_table = ds.dataset(path).to_table(
        filter=pc.field("region") == pc.scalar(region_id)
    )

    hourly_grids = hourly_grids_table.to_pandas()

    # remove other region ids from categorical region_id variable
    # hourly_grids["region"] = hourly_grids["region"].cat.remove_unused_categories()

    hourly_grids = create_date_column(hourly_grids)

    hourly_grids = fix_measurements(hourly_grids)

    daily_grids = groupby_date(hourly_grids)
    print(f"grouped by day. {daily_grids.shape}")

    daily = collapse_grid(daily_grids)
    print(f"collapsed grid to region. {daily.shape}")

    averages = calculating_moving_averages(daily)

    anomalies = calculating_anomalies(daily)

    daily_merged = pd.concat([daily, averages, anomalies], axis=1).reset_index()

    daily_merged = order_columns(daily_merged)

    # strip off the first year (not enough data for the rolling averages)
    # and the last year (which only exists because of the timezone shift from 2022-12-31:23 -> 2023-01-01:00
    daily_merged = daily_merged[
        daily_merged["date"].apply(lambda val: 1991 <= val.year <= 2022)
    ]

    save_path = TMP_PATH / (region_id + ".pqt")
    daily_merged.to_parquet(save_path)
    return save_path


def check_labeled(df):
    labeled_cols = set(ERA5_VARIABLE_VALUES)
    diff = set(df.columns).symmetric_difference(labeled_cols)
    assert not diff, diff


def main() -> pd.DataFrame:
    path = ERA5_PATH / "era5-grids.pqt"
    region_df_paths = [do_for_region(region_id, path) for region_id in track(REGIONS)]
    region_dfs = [pd.read_parquet(path) for path in region_df_paths]

    print("concat")
    regions = pd.concat(region_dfs)
    check_labeled(regions)
    regions.to_parquet(ERA5_PATH / "era5-regions-full_timeseries.pqt", index=False)

    pyreadstat.write_sav(
        df=regions,
        dst_path=ERA5_PATH / "era5-regions-full_timeseries.sav",
        column_labels=ERA5_VARIABLE_VALUES,
    )

    # Remove 2015 data, which is only needed to caclucate the averages
    regions_cut = regions[regions["date"].apply(lambda val: val.year >= 2016)]
    regions_cut.to_parquet(ERA5_PATH / "era5-regions.pqt", index=False)
    pyreadstat.write_sav(
        df=regions_cut,
        dst_path=ERA5_PATH / "era5-regions.sav",
        column_labels=ERA5_VARIABLE_VALUES,
    )


if __name__ == "__main__":
    main()
