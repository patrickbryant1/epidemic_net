#!/usr/bin/env bash
DATADIR=../../data/
RESULTSDIR=../../results/Spain/n10000/random/
N=10000
OUTDIR=../../results/Spain/n10000/random/plots/

./vis.py --datadir $DATADIR --resultsdir $RESULTSDIR --n $N --outdir $OUTDIR
