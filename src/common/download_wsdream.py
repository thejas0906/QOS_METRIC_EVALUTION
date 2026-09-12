"""
WS-DREAM Dataset Downloader.
Downloads WS-DREAM Dataset #1 (339 users x 5,825 web services) from public benchmark mirrors.
"""

import os
import sys
import time
import urllib.request
import argparse


WS_DREAM_URLS = {
    "rtmatrix.txt": "https://raw.githubusercontent.com/wronnyhuang/wsdream/master/rtMatrix.txt",
    "tpmatrix.txt": "https://raw.githubusercontent.com/wronnyhuang/wsdream/master/tpMatrix.txt",
    "userlist.txt": "https://raw.githubusercontent.com/wronnyhuang/wsdream/master/userlist.txt",
    "wslist.txt": "https://raw.githubusercontent.com/wronnyhuang/wsdream/master/wslist.txt",
}


def download_file(url: str, dest_path: str, force: bool = False):
    """Downloads a single file with progress reporting."""
    if os.path.exists(dest_path) and not force and os.path.getsize(dest_path) > 0:
        print(f"  [Skipped] {os.path.basename(dest_path)} already exists ({os.path.getsize(dest_path) / (1024*1024):.2f} MB).")
        return

    print(f"  [Downloading] {os.path.basename(dest_path)} from {url}...")
    t0 = time.time()

    def reporthook(blocknum, blocksize, totalsize):
        if blocknum % 1000 == 0 and totalsize > 0:
            pct = (blocknum * blocksize / totalsize) * 100
            print(f"    progress: {min(pct, 100):.1f}%...", end="\r", flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"  [Completed] {os.path.basename(dest_path)} ({size_mb:.2f} MB in {elapsed:.1f}s)")


def download_wsdream(target_dir: str = "data/raw/ws_dream", force: bool = False):
    """Downloads all WS-DREAM Dataset #1 files into target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    print(f"Fetching WS-DREAM Dataset #1 into '{target_dir}'...")

    for fname, url in WS_DREAM_URLS.items():
        dest = os.path.join(target_dir, fname)
        download_file(url, dest, force=force)

    print("WS-DREAM setup completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WS-DREAM Dataset #1")
    parser.add_argument("--output_dir", type=str, default="data/raw/ws_dream", help="Target output folder")
    parser.add_argument("--force", action="store_true", help="Force redownload even if files exist")
    args = parser.parse_args()

    download_wsdream(target_dir=args.output_dir, force=args.force)
