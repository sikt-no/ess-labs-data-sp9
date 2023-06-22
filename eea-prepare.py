import numpy as np
import pandas as pd
import pyreadstat

from utils import (
    EEA_PATH,
    EEA_VARIABLE_FORMATS,
    EEA_VARIABLE_LABELS,
    EEA_VARIABLE_VALUE_LABELS,
)


def clean_dates(df):
    for var in ["DatetimeBegin", "DatetimeEnd"]:
        df[var] = pd.to_datetime(df[var])
    same_day = (
        df["DatetimeBegin"].dt.date
        == (df["DatetimeEnd"] - pd.Timedelta(1, "sec")).dt.date
    )
    # exclude observations stretching over multiple days (254 for NO01 and 145 for UKI)
    df = df[same_day]
    df = df.dropna(subset="Concentration")
    df = df.sort_values("DatetimeBegin")
    print(df.shape)
    return df


def group_by_day(df):
    return (
        df.groupby(
            [
                "region",
                "AirQualityStation",
                "AirPollutant",
                pd.Grouper(key="DatetimeBegin", freq="D"),
            ]
        )
        .quantile(0.99)
        .sort_index()
        .reset_index()
    )


def group_by_region(df):
    return df.groupby(["region", "DatetimeBegin", "AirPollutant"]).max().reset_index()


def long_to_wide(df):
    return df.pivot(
        index=["region", "DatetimeBegin"],
        columns="AirPollutant",
        values="Concentration",
    ).dropna(how="all")


def calculate_n_days_sum(df, n):
    return (
        df.reset_index("region").groupby("region").rolling(f"{n}D", min_periods=n).sum()
    )


def calculate_index_variables(df_in):
    # calculate indices
    def _bin_var(name, bins):
        return (
            pd.cut(df_in[name], bins, labels=False) if name in df_in.columns else np.nan
        )

    # https://docs.google.com/document/d/1g4cxqQepVZC-5NtJ4lj4U2vrlNmf9xBR/edit
    indices = pd.DataFrame(
        {
            "aqiwdpm10": _bin_var("PM10", [0, 20, 40, 50, 100, 150, 1200]),
            "aqiwdpm2_5": _bin_var("PM2.5", [0, 10, 20, 25, 50, 75, 800]),
            "aqiwdso2": _bin_var("SO2", [0, 100, 200, 350, 500, 750, 1250]),
            "aqiwdno2": _bin_var("NO2", [0, 40, 90, 120, 230, 340, 1000]),
            "aqiwdo3": _bin_var("O3", [0, 50, 100, 130, 240, 380, 800]),
        }
    )

    # Worst air quality index level across pollutants
    indices["aqiwd"] = indices.max(axis=1)

    # Worst, last two days
    last_two_days = (
        indices.reset_index("region")
        .groupby("region")
        .rolling("2D")
        .max()
        .rename(columns=lambda col: col.replace("aqiw", "aqiw2"))
    )

    is_poor = indices >= 3
    n_days_poor_week = calculate_n_days_sum(is_poor, 7).rename(
        columns=lambda col: col.replace("aqiwd", "ndyprw")
    )
    n_days_poor_month = calculate_n_days_sum(is_poor, 30).rename(
        columns=lambda col: col.replace("aqiwd", "ndyprm")
    )
    n_days_poor_year = calculate_n_days_sum(is_poor, 365).rename(
        columns=lambda col: col.replace("aqiwd", "ndypry")
    )

    indices = pd.concat(
        [indices, last_two_days, n_days_poor_week, n_days_poor_month, n_days_poor_year],
        axis=1,
    )
    return indices


def order_columns(df):
    var_order = [
        "date",
        "region",
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
        "ndyprwpm10",
        "ndyprwpm2_5",
        "ndyprwso2",
        "ndyprwno2",
        "ndyprwo3",
        "ndyprw",
        "ndyprmpm10",
        "ndyprmpm2_5",
        "ndyprmso2",
        "ndyprmno2",
        "ndyprmo3",
        "ndyprm",
        "ndyprypm10",
        "ndyprypm2_5",
        "ndypryso2",
        "ndypryno2",
        "ndypryo3",
        "ndypry",
    ]
    # double check we are not adding or removing any columns
    diff = set(df.columns).symmetric_difference(var_order)
    assert not diff, diff
    df = df[var_order]
    return df


def main() -> pd.DataFrame:
    df_in = pd.read_parquet(EEA_PATH / "eea-stations.pqt")
    by_hour_pollutant_station = clean_dates(df_in)

    # max concentration measure pr day pr pollutant pr station (reduce on time axis)
    by_day_pollutant_station = group_by_day(by_hour_pollutant_station)

    # max concentration pr day pr pollutant pr region (reduce on geo axis)
    by_day_pollutant_region = group_by_region(by_day_pollutant_station)

    # shift from long to wide - one variable pr pollutant
    wide = long_to_wide(by_day_pollutant_region)

    indices = calculate_index_variables(wide)

    # cleanup
    indices.columns.name = None
    indices = indices.reset_index()
    # Remove 2015 data, which is only needed to caclucate rolling averages
    indices = indices[indices["DatetimeBegin"].dt.year >= 2016]
    indices["date"] = indices.pop("DatetimeBegin").dt.date

    indices = indices.convert_dtypes()
    indices = order_columns(indices)
    print(indices)
    indices.to_parquet(EEA_PATH / "eea-regions.pqt", index=False)
    pyreadstat.write_sav(
        df=indices,
        dst_path=EEA_PATH / "eea-regions.sav",
        column_labels=EEA_VARIABLE_LABELS,
        variable_value_labels=EEA_VARIABLE_VALUE_LABELS,
        variable_format=EEA_VARIABLE_FORMATS,
    )


if __name__ == "__main__":
    main()
