#!/bin/bash

BASE_DIR="/data/theorie/jjohn/git/faser_nufluxes_ml/ML_fit_enu/src/ML_fit_neutrinos/run_3_gens/fit_sibyll/fit_theta/diff_level_1_runs"

LIMIT=$1
# Loop through directories matching your naming pattern
for i in $(seq 1 $LIMIT); do
    DIR="$"runscripts_"$i"
    echo "Submitting job in $DIR"
    cd "$DIR"
    pwd
    condor_submit jobsubmit.sub 
done
