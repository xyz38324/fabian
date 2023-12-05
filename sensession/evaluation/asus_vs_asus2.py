from pathlib import Path

import polars as pl
from loguru import logger

from sensession.lib.database import Database
from sensession.evaluation.common.boxplots import create_boxplots
from sensession.evaluation.common.corr_plots import create_corrplots
from sensession.evaluation.common.preprocess import preprocess


def get_data(experiment_name: str):
    """
    Data loader (with caching to avoid repeatedly preprocessing on errors ...)
    """
    cache_path = Path.cwd() / ".cache" / "preprocessed" / experiment_name
    cache_path.mkdir(exist_ok=True, parents=True)
    cache_file = cache_path / "db.parquet"

    if cache_file.is_file():
        logger.trace("Data already preprocessed. Loading cached file ...")
        data = pl.read_parquet(cache_file)
        logger.trace(f"Loaded data of shape: {data.shape}")
        return data

    db_path = Path.cwd() / "data" / experiment_name / "db.parquet"
    logger.trace(f"Loading data from database {db_path}")
    if not db_path.is_file():
        logger.trace(f"Path {db_path} does not point to a file")

    db = Database(db_path)
    df = db.get()

    # TODO: We actually would like not to drop stuff.
    # Turns out, for an intricate experiment, we are processing structures too large for memory though (> 100 GB)
    # Therefore, we drop unused data here.
    # Ultimately this might be better suited with using a second dataframe that contains experiment ids mapped
    # to some duplicated values and employing the polars lazy API more (doesnt work well right now)
    unused_columns = [
        "experiment_start_time",
        "mask_id",
        "bandwidth",
        "tx_gain",
        "channel",
        "rssi",
        "timestamp",
        "frame_id",
        "experiment_id",
    ]
    df = df.drop(unused_columns)

    # Run preprocessing to calculate shapes etc
    data = preprocess(df)

    logger.trace(
        f"Storing preprocessed dataframe in cache ({cache_file}) for future speedup."
    )
    data.write_parquet(cache_file)

    return data


if __name__ == "__main__":
    experiment_name = "asus_vs_asus2_comparison"
    outdir = Path.cwd() / "data" / experiment_name / "images"
    outdir.mkdir(parents=True, exist_ok=True)

    data = get_data(experiment_name)
    create_corrplots(data, outdir)
    create_boxplots(data, outdir)
