#!/usr/bin/env bash
DATADIR=../../data/
RESULTSDIR=../../results/Stockholm/n100000/
N=100000 #2385643
OUTDIR=../../results/Stockholm/n100000/plots/

./vis.py --datadir $DATADIR --resultsdir $RESULTSDIR --n $N --outdir $OUTDIR
