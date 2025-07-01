#!/bin/bash

# 48-POWHEG+PYTHIA8-output-0048-W4.top
npoly=50
#PS=LHEF_analysis
PS=POWHEG+PYTHIA8-output

for ipoly in `seq ${npoly}`
do
   padpoly=`printf "%04d\n" $ipoly`
   mergedata 4 ${ipoly}-${PS}-${padpoly}-??.top -o ${ipoly}-${PS}-${padpoly}-max.top
   mergedata 5 ${ipoly}-${PS}-${padpoly}-??.top -o ${ipoly}-${PS}-${padpoly}-min.top
   cp ${ipoly}-${PS}-${padpoly}-W1.top ${ipoly}-${PS}-${padpoly}-11.top
   for hist in '11' 'min' 'max'
   do
      for index in `grep index ${ipoly}-${PS}-${padpoly}-${hist}.top | awk '{print $2}'`
      do
         pastegnudata "[$index]" ${ipoly}-${PS}-${padpoly}-${hist}.top > ${ipoly}-${PS}-${padpoly}-${hist}-${index}.dat
         sed -i 's/D/E/g' ${ipoly}-${PS}-${padpoly}-${hist}-${index}.dat
      done
   done
done

         
