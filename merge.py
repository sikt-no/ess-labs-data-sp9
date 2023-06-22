import datetime as dt
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyreadstat

from utils import (
    EEA_PATH,
    EEA_VARIABLE_FORMATS,
    EEA_VARIABLE_LABELS,
    EEA_VARIABLE_VALUE_LABELS,
    ERA5_PATH,
    ERA5_VARIABLE_VALUES,
    ESS_FILES,
)


def merge_round(ess_filename: str, eea_table: pa.Table, era5_table: pa.Table):
    df, md = pyreadstat.read_sav(ess_filename)
    print(f"read {ess_filename}, {df.shape}")
    if "inwds" in df.columns:
        df["interview_date"] = df["inwds"].dt.date
    elif "questcmp" in df.columns:
        df["interview_date"] = df["questcmp"].dt.date
    else:
        df = df.dropna(subset=["inwdds", "inwmms", "inwyys"])
        df["interview_date"] = df.apply(
            lambda row: dt.date(
                year=int(row["inwyys"]),
                month=int(row["inwmms"]),
                day=int(row["inwdds"]),
            ),
            axis=1,
        )
    ess_table = pa.Table.from_pandas(df)
    print(f"ESS, {ess_table.num_rows}x{ess_table.num_columns}")
    print(ess_table["region"].value_counts())

    ess_eea = ess_table.join(
        eea_table,
        keys=["region", "interview_date"],
        right_keys=["region", "date"],
        join_type="inner",
    )
    print(f"merged, {ess_eea.num_rows}x{ess_eea.num_columns}")
    print(ess_eea["region"].value_counts())

    ess_eea_era5 = ess_eea.join(
        era5_table,
        keys=["region", "interview_date"],
        right_keys=["region", "date"],
        join_type="inner",
    )
    print(f"merged, {ess_eea_era5.num_rows}x{ess_eea_era5.num_columns}")
    print(ess_eea_era5["region"].value_counts())

    path = Path("merged-EOSC-" + ess_filename).with_suffix(".pqt")
    pq.write_table(ess_eea_era5, path)

    as_df = ess_eea_era5.to_pandas()
    try:
        as_df = as_df.drop(columns="__index_level_0__")
    except KeyError:
        pass
    column_labels = {
        **md.column_names_to_labels,
        **EEA_VARIABLE_LABELS,
        **ERA5_VARIABLE_VALUES,
    }
    variable_value_labels = {
        **md.variable_value_labels,
        **EEA_VARIABLE_VALUE_LABELS,
    }
    variable_formats = {
        **EEA_VARIABLE_FORMATS,
        **md.original_variable_types,
    }
    pyreadstat.write_sav(
        df=as_df,
        dst_path=path.with_suffix(".sav"),
        column_labels=column_labels,
        variable_value_labels=variable_value_labels,
        variable_display_width=md.variable_display_width,
        variable_measure=md.variable_measure,
        missing_ranges=md.missing_ranges,
        variable_format=variable_formats,
    )
    print()


def main():
    eea_table = pq.read_table(EEA_PATH / "eea-regions.pqt")
    print(f"read eea_table, {eea_table.num_rows}x{eea_table.num_columns}")
    print(eea_table["region"].value_counts())
    print()

    era5_table = pq.read_table(ERA5_PATH / "era5-regions.pqt")
    print(f"read era5_table, {era5_table.num_rows}x{era5_table.num_columns}")
    print(era5_table["region"].value_counts())
    print()

    for filename in ESS_FILES:
        merge_round(filename, eea_table, era5_table)


if __name__ == "__main__":
    main()
