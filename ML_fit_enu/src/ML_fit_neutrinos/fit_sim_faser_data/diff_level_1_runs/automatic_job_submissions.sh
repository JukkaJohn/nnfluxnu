#!/bin/bash

BASE_DIR="/data/theorie/jjohn/git/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos/fit_sim_faser_data/diff_level_1_runs"

LIMIT=$1
# Loop through directories matching your naming pattern
for i in $(seq 1 $LIMIT); do
    DIR="$BASE_DIR/runscripts_$i"
    echo "Submitting job in $DIR"
    cd "$DIR"
    pwd
    condor_submit jobsubmit.sub 
done
