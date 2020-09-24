#!/usr/bin/env bash
DATADIR=../../data/
RESULTSDIR=../../results/New_York/n10000/preferential_attachment/
N=10000 #2385643
OUTDIR=../../results/New_York/n10000/preferential_attachment/plots/

./vis.py --datadir $DATADIR --resultsdir $RESULTSDIR --n $N --outdir $OUTDIR
