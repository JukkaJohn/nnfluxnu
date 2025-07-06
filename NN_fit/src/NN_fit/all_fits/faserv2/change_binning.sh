#!/bin/bash

# List of target filenames
TARGET_FILES=("fit_dpmjet.py" "fit_sibyll.py" "fit_qgsjet.py" "fit_epos.py" "postfit_analysis.py" "postfit_analysis_level1_instances.py")

# Loop through each target filename
for FILENAME in "${TARGET_FILES[@]}"; do
  echo "Searching for $FILENAME..."

  # Use 'find' to search for matching files recursively
  find . -type f -name "$FILENAME" | while read -r FILE; do
    echo "Updating $FILE..."
    # Replace all occurrences of 'final' with 'fine'
    sed -i '' 's/FK_Enu_binsize/FK_Enu_fine_binsize/g' "$FILE"
  done
done

echo "All replacements complete."

