#!/bin/bash

prg=../pwhg_main
npoly=50
ncores=50

function limit_procs {
    while [ `jobs -p | wc -w` -ge $ncores ]
    do
	sleep 15
    done
}

cp powheg.input-save powheg.input
for ipoly in `seq ${npoly}`
do
   echo ${ipoly}
   padipoly=`printf "%04d\n" $ipoly`
   echo ${ipoly}-events-${padipoly}.lhe | ../main-PYTHIA8-lhef > run-${padipoly}.log &
   #echo ${ipoly}-events-${padipoly}.lhe | ../lhef_analysis > run-${padipoly}.log &
   limit_procs
done
wait
