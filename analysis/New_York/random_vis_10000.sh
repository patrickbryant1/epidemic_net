#!/usr/bin/env bash
DATADIR=../../data/
RESULTSDIR=../../results/New_York/n10000/random/
N=10000 #2385643
OUTDIR=../../results/New_York/n10000/random/plots/

./vis.py --datadir $DATADIR --resultsdir $RESULTSDIR --n $N --outdir $OUTDIR
