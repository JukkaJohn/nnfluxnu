#!/bin/bash

prg=../pwhg_main
npoly=5
ncores=5

function limit_procs {
    while [ `jobs -p | wc -w` -ge $ncores ]
    do
	sleep 15
    done
}

for ipoly in `seq $npoly`
do
   echo ${ipoly}
   cat powheg.input-save | sed "s/ipoly.*/ipoly ${ipoly}/" > ${ipoly}-powheg.input
   echo ${ipoly} | ${prg} > run-poly-${ipoly}.log &
   limit_procs
done
wait
