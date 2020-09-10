#!/usr/bin/env bash
DATADIR=./data/
N=10000 #2385643
M=1
SR=0.5 #Spread reduction (multiplied with infection probability) - this should be introduced on the day of intervention
OUTDIR=./results/n10000/
for M in 1 2 3 5 7 10
  do
    for SR in 0.1 0.2 0.3 0.4
      do
        ./model.py --datadir $DATADIR --n $N --m $M --s $SR --outdir $OUTDIR
      done
  done
