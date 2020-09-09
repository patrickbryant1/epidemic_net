#!/usr/bin/env bash
RESULTDF=./results/n10000/results.csv
N=10000 #2385643
OUTDIR=./results/n10000/
./vis.py --resultdf $RESULTDF --n $N --outdir $OUTDIR
