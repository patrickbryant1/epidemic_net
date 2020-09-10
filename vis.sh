#!/usr/bin/env bash
DATADIR=./data/
RESULTDF=./results/n10000/results.csv
N=10000 #2385643
OUTDIR=./results/n10000/plots/
for M in 1 2 3 5 7 10
  do
    for SR in '2_2_2_2_2_2' '2_2_2_3_3_3' '2_2_2_4_4_4'
      do
        ./vis.py --datadir $DATADIR --resultdf ./results/n10000/results_$M'_'$SR'.csv' --n $N --m $M --s $SR --outdir $OUTDIR
      done
  done
