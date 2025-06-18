#!/bin/bash

# Find and modify postfit_analysis.py and fit_dpmjet.py in subdirectories
find . -type f \( -name "postfit_analysis.py" -o -name "fit_dpmjet.py" \) | while read -r file; do
    echo "Modifying $file..."
    # Replace 'final' with 'fine' in FK filenames
    sed -i 's/FK_\([^"]*\)_final/FK_\1_fine/g' "$file"

    # Replace FK_{obs}_binsize.dat with FK_{obs}_fine_binsize.dat
    sed -i 's/FK_\([^"]*\)_binsize.dat/FK_\1_fine_binsize.dat/g' "$file"
done

echo "Replacement complete."

