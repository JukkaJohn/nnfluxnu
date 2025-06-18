#!/bin/bash

# This script replaces all instances of 'final' with 'final' (or another word) in fit_dpmjet.py files.

# Change the replacement target if needed
SEARCH="final"
REPLACE="final"  # Change this if you want to actually replace it with something else

find . -type f -name "fit_epos.py" | while read -r file; do
    echo "Processing $file"
    sed -i "s/${SEARCH}/${REPLACE}/g" "$file"
done

