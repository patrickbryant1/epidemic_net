#!/usr/bin/env bash
DATADIR=./data/
N=1000 #2385643
M=5
OUTDIR=./results/n10000/
./model.py --datadir $DATADIR --n $N --m $M --outdir $OUTDIR
