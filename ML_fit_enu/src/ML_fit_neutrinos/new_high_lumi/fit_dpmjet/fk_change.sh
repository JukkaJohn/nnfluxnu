#!/bin/bash

# This script finds and replaces all instances of 'final' with 'final' in fit_dpmjet.py in subdirectories.

# Find all files named fit_dpmjet.py in subdirectories
find . -type f -name "fit_dpmjet.py" | while read -r file; do
    echo "Processing $file"
    # Replace all 'final' with 'final' using sed (a no-op replacement)
    sed -i 's/\bfinal\b/final/g' "$file"
done
