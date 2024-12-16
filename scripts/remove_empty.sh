#!/bin/bash

# Usage: ./remove_empty_folders.sh [directory]
# If no directory is specified, it defaults to the current directory.

# Get the target directory (default is current directory)
TARGET_DIR=${1:-.}

# Find and remove empty directories
find "$TARGET_DIR" -type d -empty -delete

echo "Empty folders removed from $TARGET_DIR."