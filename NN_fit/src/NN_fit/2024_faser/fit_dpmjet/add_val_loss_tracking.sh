#!/bin/bash

# Set the root directory to search (change as needed or pass as $1)
ROOT_DIR="/Users/jukkajohn/Masterscriptie/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos/2024_faser/fit_dpmjet"

# Code to append
CODE=$'if validation != 0.0:\n    with open("../validation_losses.txt", "a") as f:\n        np.savetxt(f, ["new run"], fmt="%s", delimiter=",")\n        np.savetxt(f, validation_losses, delimiter=",")'

# Find and modify each fit_dpmjet.py file
find "$ROOT_DIR" -type f -name "fit_dpmjet.py" | while read -r file; do
    echo "Appending code to: $file"
    echo -e "\n$CODE" >> "$file"
done
