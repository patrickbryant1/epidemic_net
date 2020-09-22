#!/usr/bin/env bash
DATADIR=../../data/
RESULTSDIR=../../results/Spain/n10000/
N=10000
OUTDIR=../../results/Spain/n10000/plots/

./vis.py --datadir $DATADIR --resultsdir $RESULTSDIR --n $N --outdir $OUTDIR
