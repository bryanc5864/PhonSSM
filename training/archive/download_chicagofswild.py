"""
ChicagoFSWild Dataset Downloader
Downloads the ChicagoFSWild fingerspelling dataset (14 GB).

Source: https://home.ttic.edu/~klivescu/ChicagoFSWild.htm

Usage:
    python training/download_chicagofswild.py
    python training/download_chicagofswild.py --download-only
"""

import subprocess
import tarfile
from pathlib import Path
from tqdm import tqdm
import argparse

CHICAGOFSWILD_URL = "https://dl.ttic.edu/ChicagoFSWild.tgz"
DATA_RAW = Path(__file__).parent.parent / "data" / "raw"


def download_chicagofswild(data_dir=None, download_only=False):
    """Download and extract ChicagoFSWild dataset."""
    if data_dir is None:
        data_dir = DATA_RAW

    data_dir = Path(data_dir)
    archive_path = data_dir / "chicagofswild.tgz"
    extract_dir = data_dir / "chicagofswild"

    print("=" * 60)
    print("CHICAGOFSWILD DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Size: 14 GB")
    print(f"Sequences: 7,304")
    print(f"Signers: 160")
    print("=" * 60)

    # Download
    if archive_path.exists():
        print(f"\nArchive already exists: {archive_path}")
    else:
        print(f"\nDownloading from {CHICAGOFSWILD_URL}...")
        data_dir.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                ['curl', '-L', '-C', '-', '-o', str(archive_path), CHICAGOFSWILD_URL],
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("curl failed, trying wget...")
            try:
                subprocess.run(
                    ['wget', '-c', '-O', str(archive_path), CHICAGOFSWILD_URL],
                    check=True
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("ERROR: Could not download. Install curl or wget.")
                return False

    if download_only:
        print(f"\nDownload complete: {archive_path}")
        return True

    # Extract
    if extract_dir.exists() and list(extract_dir.rglob("*.csv")):
        print(f"\nAlready extracted: {extract_dir}")
    else:
        print(f"\nExtracting to {extract_dir}...")
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                members = tar.getmembers()
                for member in tqdm(members, desc="Extracting"):
                    tar.extract(member, extract_dir)
            print("Extraction complete!")
        except Exception as e:
            print(f"ERROR extracting: {e}")
            return False

    print(f"\nDataset ready at: {extract_dir}")
    print("\nTo process, run:")
    print("  python training/process_fingerspelling.py")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download ChicagoFSWild dataset')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory to save raw data')
    parser.add_argument('--download-only', action='store_true',
                        help='Only download, do not extract')

    args = parser.parse_args()
    download_chicagofswild(args.data_dir, args.download_only)
