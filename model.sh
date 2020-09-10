#!/usr/bin/env bash
DATADIR=./data/
N=10000 #2385643
M=1
SR=0.5 #Spread reduction (multiplied with infection probability) - this should be introduced on the day of intervention
OUTDIR=./results/n10000/
for M in 1 #2 3 5
  do
    for SR in '1_1_1_1_1_1' #'2_2_2_2_2_2' '2_2_3_3_3_3' '2_2_4_4_4_4' '2_2_5_5_5_5'
      do
        ./model.py --datadir $DATADIR --n $N --m $M --s $SR --outdir $OUTDIR
      done
  done
