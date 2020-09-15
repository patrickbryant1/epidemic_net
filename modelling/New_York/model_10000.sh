#!/usr/bin/env bash
DATADIR=../../data/
N=10000 #2385643
#M = number of new links to introduce
#SR = Spread reduction (multiplied with infection probability) - this should be introduced on the day of intervention
NINITIAL=1 #How many nodes to pick in the initial step
PC=1 #Pseudo count
OUTDIR=../../results/New_York/n10000/
for M in 1 2 3 4 5
  do
    for SR in '1_1_1_1' '2_2_2_2' '4_4_4_4' '1_1_2_2' '1_1_4_4' '2_2_1_1' '4_4_1_1'
      do
        ./model.py --datadir $DATADIR --n $N --m $M --s $SR --num_initial $NINITIAL --pseudo_count $PC --outdir $OUTDIR
      done
  done
