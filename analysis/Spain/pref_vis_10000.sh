#!/usr/bin/env bash
DATADIR=../../data/
RESULTSDIR=../../results/Spain/n10000/preferential_attachment/
N=10000
OUTDIR=../../results/Spain/n10000/preferential_attachment/plots/

./vis.py --datadir $DATADIR --resultsdir $RESULTSDIR --n $N --outdir $OUTDIR
