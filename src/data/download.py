"""
Data Acquisition Module for Customer Support Twitter Dataset.
Supports:
1. Direct streaming / sample download from verified mirror with byte range or streaming chunks.
2. Kaggle CLI download (if credentials exist).
3. Local CSV copying / symlinking without hardcoding paths.
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("src.data.download")

DEFAULT_DATASET_URL = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_SAMPLE_DEST = DEFAULT_RAW_DIR / "twcs_sample.csv"


def download_sample_from_url(
    url: str = DEFAULT_DATASET_URL,
    output_path: Path = DEFAULT_SAMPLE_DEST,
    sample_size: int = 30000,
    chunk_size_bytes: int = 1024 * 1024,
) -> Path:
    """
    Streams CSV rows from remote repository until approximately `sample_size` records are collected.
    Ensures that only complete CSV lines are preserved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Initiating streaming download from %s (target ~%d rows)...", url, sample_size)

    # Use streaming GET request
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        
        lines_written = 0
        buffer = ""
        with open(output_path, "w", encoding="utf-8", errors="replace") as out_f:
            for chunk in resp.iter_content(chunk_size=chunk_size_bytes, decode_unicode=True):
                if not chunk:
                    continue
                buffer += chunk
                lines = buffer.splitlines(keepends=True)
                
                # Keep the last line in the buffer if it might be incomplete
                if not buffer.endswith("\n") and not buffer.endswith("\r"):
                    buffer = lines.pop() if lines else ""
                else:
                    buffer = ""
                
                for line in lines:
                    out_f.write(line)
                    lines_written += 1
                    if lines_written >= sample_size + 1:  # +1 for header
                        break
                
                if lines_written >= sample_size + 1:
                    break

    logger.info("Saved %d rows to %s", lines_written - 1, output_path)
    return output_path


def copy_local_csv(input_path: Path, output_path: Path = DEFAULT_SAMPLE_DEST, sample_size: Optional[int] = None) -> Path:
    """Copies or samples from a locally provided CSV path."""
    if not input_path.exists():
        raise FileNotFoundError(f"Provided input file does not exist: {input_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if sample_size is None:
        logger.info("Copying full file from %s to %s", input_path, output_path)
        shutil.copyfile(input_path, output_path)
    else:
        logger.info("Sampling %d rows from %s to %s", sample_size, input_path, output_path)
        with open(input_path, "r", encoding="utf-8", errors="replace") as src, \
             open(output_path, "w", encoding="utf-8") as dst:
            for idx, line in enumerate(src):
                dst.write(line)
                if idx >= sample_size:
                    break
    return output_path


def download_via_kaggle(output_path: Path = DEFAULT_SAMPLE_DEST) -> Path:
    """Attempts to download dataset using Kaggle CLI."""
    import subprocess
    logger.info("Attempting Kaggle CLI download: thoughtvector/customer-support-on-twitter")
    raw_dir = output_path.parent
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", "thoughtvector/customer-support-on-twitter", "-p", str(raw_dir), "--unzip"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle CLI failed: {result.stderr}")
    
    # Locate twcs.csv
    twcs_file = raw_dir / "twcs.csv"
    if twcs_file.exists():
        logger.info("Successfully downloaded and unzipped %s", twcs_file)
        return twcs_file
    raise FileNotFoundError("Could not locate twcs.csv after Kaggle download")


def parse_args():
    parser = argparse.ArgumentParser(description="Acquire and sample the Customer Support Twitter dataset.")
    parser.add_argument("--sample-size", type=int, default=int(os.getenv("SAMPLE_SIZE", "30000")),
                        help="Number of lines to sample (default: 30000).")
    parser.add_argument("--input-path", type=str, default=None,
                        help="Path to a locally available twcs.csv file.")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_SAMPLE_DEST),
                        help="Path where sampled raw CSV will be saved.")
    parser.add_argument("--source-url", type=str, default=DEFAULT_DATASET_URL,
                        help="Remote URL mirror for twcs.csv.")
    parser.add_argument("--use-kaggle", action="store_true",
                        help="Attempt download using Kaggle CLI.")
    return parser.parse_args()


def main():
    args = parse_args()
    dest = Path(args.output_path)
    
    if args.input_path:
        copy_local_csv(Path(args.input_path), dest, sample_size=args.sample_size)
    elif args.use_kaggle:
        try:
            download_via_kaggle(dest)
        except Exception as e:
            logger.warning("Kaggle download failed (%s). Falling back to direct URL streaming.", e)
            download_sample_from_url(args.source_url, dest, args.sample_size)
    else:
        download_sample_from_url(args.source_url, dest, args.sample_size)

    # Verify output
    if dest.exists() and dest.stat().st_size > 0:
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info("Dataset ready at %s (Size: %.2f MB)", dest, size_mb)
    else:
        logger.error("Dataset download failed or output file is empty.")
        sys.exit(1)


if __name__ == "__main__":
    main()
