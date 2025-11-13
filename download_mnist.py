#!/usr/bin/env python3
"""
Helper script to download MNIST dataset from alternative mirrors.

The original MNIST hosting (yann.lecun.com) is often unavailable.
This script downloads from reliable alternative sources.
"""

import os
import urllib.request
import gzip
import shutil
from pathlib import Path


def download_file(url, destination):
    """Download a file with progress indication."""
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"  ✓ Saved to {destination}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_mnist(data_dir='./data'):
    """
    Download MNIST dataset from alternative mirrors.

    Args:
        data_dir: Directory to save MNIST data
    """
    # Create directories
    raw_dir = Path(data_dir) / 'MNIST' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Alternative mirrors for MNIST
    mirrors = [
        'https://ossci-datasets.s3.amazonaws.com/mnist',  # PyTorch mirror
        'https://storage.googleapis.com/cvdf-datasets/mnist',  # Google mirror
    ]

    # MNIST files
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
    ]

    print(f"\nDownloading MNIST dataset to {raw_dir}\n")

    # Try each mirror
    for mirror in mirrors:
        print(f"Trying mirror: {mirror}")
        success_count = 0

        for filename in files:
            destination = raw_dir / filename

            # Skip if already exists
            if destination.exists():
                print(f"  ✓ {filename} already exists")
                success_count += 1
                continue

            # Try to download
            url = f"{mirror}/{filename}"
            if download_file(url, destination):
                success_count += 1

        if success_count == len(files):
            print(f"\n✓ Successfully downloaded all MNIST files!")
            return True

        print(f"  ({success_count}/{len(files)} files downloaded from this mirror)\n")

    # Check final status
    existing_files = sum(1 for f in files if (raw_dir / f).exists())
    if existing_files == len(files):
        print(f"\n✓ All MNIST files are present!")
        return True
    else:
        print(f"\n✗ Missing {len(files) - existing_files} files")
        print("\nManual download option:")
        print("You can manually download MNIST from:")
        print("  https://github.com/pytorch/vision/tree/main/torchvision/datasets")
        print(f"\nPlace the .gz files in: {raw_dir}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download MNIST dataset')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to save MNIST data (default: ./data)')

    args = parser.parse_args()

    success = download_mnist(args.data_dir)
    exit(0 if success else 1)
