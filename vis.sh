#!/usr/bin/env bash
DATADIR=./data/
RESULTDF=./results/n10000/results.csv
N=10000 #2385643
OUTDIR=./results/n10000/
for M in 1 2 3 5 7 10
  do
    for SR in 0.1 0.2 0.3 0.4
      do
        ./vis.py --datadir $DATADIR --resultdf ./results/n10000/results_$M'_'$SR'.csv' --n $N --m $M --s $SR --outdir $OUTDIR
      done
  done
