#!/usr/bin/env python3
"""
Script to copy image files from REALM Paper folder to results/paper_figures
"""

import os
import shutil
from pathlib import Path

# Define source and destination directories
source_dir = Path("REALM Paper (arXiv)")
dest_dir = Path("results/paper_figures")

# Create destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Get all PNG files from source directory
png_files = list(source_dir.glob("*.png"))

# Copy each PNG file to destination
for png_file in png_files:
    dest_file = dest_dir / png_file.name
    print(f"Copying {png_file} to {dest_file}")
    shutil.copy2(png_file, dest_file)

print(f"Copied {len(png_files)} image files to {dest_dir}")