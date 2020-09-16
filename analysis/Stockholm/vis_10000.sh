#!/usr/bin/env bash
DATADIR=../../data/
RESULTSDIR=../../results/Stockholm/n10000/
N=10000 #2385643
OUTDIR=../../results/Stockholm/n10000/plots/

./vis.py --datadir $DATADIR --resultsdir $RESULTSDIR --n $N --outdir $OUTDIR
