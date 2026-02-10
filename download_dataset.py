#!/usr/bin/env python3
"""
Dataset downloader for CRISP training
Downloads and prepares popular image enhancement datasets
"""

import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path

def download_file(url, filename):
    """Download file with progress"""
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            print(f"\rDownloading {filename}: {percent:.1f}%", end='', flush=True)
    
    urllib.request.urlretrieve(url, filename, progress_hook)
    print(f"\n✅ Downloaded: {filename}")

def extract_archive(filename, extract_to):
    """Extract zip/tar files"""
    print(f"Extracting {filename}...")
    
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif filename.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(filename, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    
    print(f"✅ Extracted to: {extract_to}")

def download_lol_dataset():
    """Download LOL (Low-Light) dataset"""
    print("📥 Downloading LOL Dataset...")
    
    os.makedirs('datasets/LOL', exist_ok=True)
    
    # LOL dataset URLs (example - replace with actual URLs)
    urls = {
        'train_low.zip': 'https://example.com/LOL/train_low.zip',
        'train_high.zip': 'https://example.com/LOL/train_high.zip',
        'test_low.zip': 'https://example.com/LOL/test_low.zip',
        'test_high.zip': 'https://example.com/LOL/test_high.zip'
    }
    
    for filename, url in urls.items():
        try:
            download_file(url, f'datasets/LOL/{filename}')
            extract_archive(f'datasets/LOL/{filename}', 'datasets/LOL/')
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
    
    print("✅ LOL Dataset ready!")

def download_fivek_preprocessed():
    """Download preprocessed FiveK dataset"""
    print("📥 Downloading FiveK Dataset (Preprocessed)...")
    
    os.makedirs('datasets/FiveK', exist_ok=True)
    
    # This is a placeholder - actual FiveK requires registration
    print("⚠️  FiveK requires manual download from:")
    print("   https://data.csail.mit.edu/graphics/fivek/")
    print("   Or preprocessed version from:")
    print("   https://github.com/HuiZeng/Image-Adaptive-3DLUT")

def create_sample_dataset():
    """Create a small sample dataset for testing"""
    print("📥 Creating sample dataset...")
    
    # Create directory structure
    dirs = [
        'datasets/sample/train/lq',
        'datasets/sample/train/hq',
        'datasets/sample/val/lq', 
        'datasets/sample/val/hq'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✅ Sample dataset structure created!")
    print("📁 Add your images to:")
    for dir_path in dirs:
        print(f"   {dir_path}/")

def download_dataset(dataset_name):
    """Download specific dataset"""
    
    datasets = {
        'lol': download_lol_dataset,
        'fivek': download_fivek_preprocessed,
        'sample': create_sample_dataset
    }
    
    if dataset_name.lower() in datasets:
        datasets[dataset_name.lower()]()
    else:
        print(f"❌ Unknown dataset: {dataset_name}")
        print("Available datasets:", list(datasets.keys()))

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download datasets for CRISP')
    parser.add_argument('dataset', choices=['lol', 'fivek', 'sample'], 
                       help='Dataset to download')
    
    args = parser.parse_args()
    download_dataset(args.dataset)