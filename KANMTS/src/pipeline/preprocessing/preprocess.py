from os import path, remove
from pathlib import Path
from tkinter.ttk import Separator
from functools import reduce
import polars as pl
import os

def get_files(root_folder: str) -> list[Path]:
    folder = Path(root_folder)
    return [f for f in folder.iterdir() if f.is_file()]

def load_files(files: list[Path]) -> list[pl.LazyFrame]:
    return [pl.scan_csv(str(f), separator='\t', infer_schema_length=0) for f in files]

def clean_cols(lfs: list[pl.LazyFrame]) -> list[pl.LazyFrame]:
    return [
        lf.select(pl.all().name.replace(r"[<>]", ""))
        for lf in lfs
    ]

def concat_time_cols(lfs: list[pl.LazyFrame]) -> list[pl.LazyFrame]:
    return [
        lf.with_columns(
            pl.concat_str([pl.col("DATE"), pl.col("TIME")], separator=" ").alias("DATE")
        ).drop("TIME")
        for lf in lfs
    ]

def add_prefix(lfs: list[pl.LazyFrame], files: list[Path]) -> list[pl.LazyFrame]:
    return [
        lf.select(
            pl.col("DATE"),
            pl.all().exclude("DATE").name.prefix(f"{f.stem}_")
        )
        for lf, f in zip(lfs, files)
    ]

def save_data(lf : pl.LazyFrame, output_folder : str, filename : str = "output") -> None:
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)

    file_path = os.path.join(output, f'{filename}.csv')
    lf.collect().write_csv(file_path, separator=',')
    

def concat_lfs(lfs: list[pl.LazyFrame]) -> pl.LazyFrame:
    return reduce(
        lambda acc, lf: acc.join(lf, on="DATE", how="full", coalesce=True),
        lfs
    )

def remove_useless_cols(lfs : list[pl.LazyFrame], name1 : str, name2 :str) -> list[pl.LazyFrame]:
    return [lf.drop(name1, name2) for lf in lfs]

def fill_time_gaps(df : pl.DataFrame) -> pl.DataFrame:
    df_sorted = df.sort("DATE")
    df_filled = df_sorted.upsample(time_column="DATE", every="15m")
    return df_filled

def apply_clean_borders(lf : pl.LazyFrame) -> pl.LazyFrame:
    lf_i = lf.with_row_index("row_nr")

    started = pl.all_horizontal(pl.all().exclude("DATE", "row_nr").is_not_null())

    first_row = (
        lf_i.filter(started)
            .select(pl.col("row_nr").min())
            .collect()
            .item()
    )
    last_row = (
        lf_i.filter(started)
            .select(pl.col("row_nr").max())
            .collect()
            .item()
    )
    return lf_i.filter((pl.col("row_nr") >= first_row) & (pl.col("row_nr") <= last_row)).drop("row_nr")

def apply_linear_interpolate(lf : pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        pl.all().exclude("DATE")
        .cast(pl.Float64)
        .interpolate(method='linear')
    )


def process_data(root_folder: str, output_folder: str) -> None:
    files = get_files(root_folder)
    lfs   = load_files(files)
    lfs   = clean_cols(lfs)
    lfs   = concat_time_cols(lfs)
    lfs   = remove_useless_cols(lfs, 'SPREAD', 'VOL')
    lfs   = add_prefix(lfs, files)
    lf    = concat_lfs(lfs)
    lf    = apply_clean_borders(lf)
    lf    = apply_linear_interpolate(lf)
    save_data(lf, output_folder)
    


