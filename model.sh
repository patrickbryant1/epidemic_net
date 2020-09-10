#!/usr/bin/env bash
DATADIR=./data/
N=10000 #2385643
M=1
OUTDIR=./results/n10000/
for M in 1 2 3 5 7 10 15
  do
    ./model.py --datadir $DATADIR --n $N --m $M --outdir $OUTDIR
done
