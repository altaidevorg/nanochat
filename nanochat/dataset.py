"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import glob
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
# add configuration for your own if needed
ds_infos = {
    "karpathy": {
        "BASE_URL": "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main",
        "MAX_SHARD": 1822,  # the last datashard is shard_01822.parquet
    },
    "altai": {
        "BASE_URL": "https://huggingface.co/datasets/altaidevorg/fineweb2-hq-turkish/resolve/main",
        "MAX_SHARD": 90,
    },
}

index_to_filename = (
    lambda index: f"shard_{index:05d}.parquet"
)  # format of the filenames
base_dir = get_base_dir()
MAIN_DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(MAIN_DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported


# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported


def list_parquet_files(dataset_name=None, data_dir=None):
    """
    Looks into a data dir (or dataset subfolder) and returns parquet files.
    If dataset_name is specified, returns parquets for that dataset.
    If dataset_name is None, checks for dataset subdirectories or returns all parquets.
    """
    if data_dir is None:
        if dataset_name:
            data_dir = os.path.join(MAIN_DATA_DIR, dataset_name)
        else:
            data_dir = MAIN_DATA_DIR

    if dataset_name is None and os.path.exists(data_dir):
        # Check if there are dataset subdirectories (e.g. karpathy/, altai/)
        subdirs = [
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]
        if subdirs:
            result = {}
            for sd in sorted(subdirs):
                files = sorted(glob.glob(f"{os.path.join(data_dir, sd)}/*.parquet"))
                if files:
                    result[sd] = files
            if result:
                return result

    # Single directory case
    parquet_paths = sorted(glob.glob(f"{data_dir}/**/*.parquet"))
    return parquet_paths


def parquets_iter_batched(split, dataset_name=None, start=0, step=1):
    """
    Iterate through a dataset, yielding batches of underlying row_groups for efficiency.
    - split can be "train" or "val". The last parquet file of the dataset will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(dataset_name=dataset_name)
    if isinstance(parquet_paths, dict):
        # If multiple datasets found and none specified, flatten or pick first
        if dataset_name in parquet_paths:
            parquet_paths = parquet_paths[dataset_name]
        else:
            flattened = []
            for paths in parquet_paths.values():
                flattened.extend(paths)
            parquet_paths = sorted(flattened)

    if not parquet_paths:
        return

    if split == "train":
        parquet_paths = parquet_paths[:-1] if len(parquet_paths) > 1 else parquet_paths
    else:
        parquet_paths = parquet_paths[-1:]

    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            yield texts


def interleaved_parquets_iter_batched(
    split, datasets=None, weights=None, start=0, step=1, seed=42
):
    """
    Weighted interleaving stream across multiple dataset iterators.
    - datasets: list of dataset names (e.g. ['karpathy', 'altai']) or dict/None (autodetect).
    - weights: relative weights for sampling each dataset (e.g. [0.7, 0.3]).
    - start/step: DDP rank and world_size.
    - seed: random seed for reproducible weighted sampling across ranks.
    """
    import random

    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # Discover available datasets if not explicitly provided
    if datasets is None:
        all_files = list_parquet_files()
        if isinstance(all_files, dict):
            datasets = list(all_files.keys())
        else:
            datasets = [None]
    elif isinstance(datasets, str):
        datasets = [ds.strip() for ds in datasets.split(",")]

    if not datasets:
        return

    # Parse and normalize weights
    if weights is None:
        weights = [1.0] * len(datasets)
    elif isinstance(weights, str):
        weights = [float(w.strip()) for w in weights.split(",")]
    elif isinstance(weights, (int, float)):
        weights = [float(weights)]

    assert len(datasets) == len(
        weights
    ), f"Number of datasets ({len(datasets)}) must match weights ({len(weights)})"

    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]

    # Helper function to create an infinite generator for a single dataset
    def infinite_dataset_gen(ds_name):
        while True:
            has_data = False
            for batch in parquets_iter_batched(
                split=split, dataset_name=ds_name, start=start, step=step
            ):
                has_data = True
                yield batch
            if not has_data:
                break

    # Instantiate generators for active datasets
    generators = {}
    active_datasets = []
    active_weights = []
    for ds_name, w in zip(datasets, norm_weights):
        gen = infinite_dataset_gen(ds_name)
        generators[ds_name] = gen
        active_datasets.append(ds_name)
        active_weights.append(w)

    if not active_datasets:
        return

    # Deterministic PRNG per rank
    rng = random.Random(seed + start)

    while True:
        # Sample dataset according to weights
        selected_ds = rng.choices(active_datasets, weights=active_weights, k=1)[0]
        try:
            batch = next(generators[selected_ds])
            yield batch
        except StopIteration:
            # If dataset iterator exhausted completely and produced no items, remove it
            idx = active_datasets.index(selected_ds)
            active_datasets.pop(idx)
            active_weights.pop(idx)
            if not active_datasets:
                break



# -----------------------------------------------------------------------------
def download_single_file(args):
    dataset_name, index = args
    """ Downloads a single file index, with some backoff """
    ds_info = ds_infos[dataset_name]
    BASE_URL = ds_info["BASE_URL"]
    DATA_DIR = os.path.join(MAIN_DATA_DIR, dataset_name)

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2**attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu 100BT dataset shards"
    )
    parser.add_argument(
        "-d", "--dataset", choices=("karpathy", "altai"), default="karpathy"
    )
    parser.add_argument(
        "-n",
        "--num-files",
        type=int,
        default=-1,
        help="Number of shards to download (default: -1), -1 = disable",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )
    args = parser.parse_args()

    ds_info = ds_infos[args.dataset]
    MAX_SHARD = ds_info["MAX_SHARD"]
    DATA_DIR = os.path.join(MAIN_DATA_DIR, args.dataset)
    os.makedirs(DATA_DIR, exist_ok=True)

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = [(args.dataset, index) for index in range(num)]
    print(
        f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers..."
    )
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
