#!/bin/bash

# Run fit_sim_data.py 5 times
for i in {1..20}
do
    echo "Run #$i"
    python fit_dpmjet.py
done
