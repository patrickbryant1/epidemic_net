#!/usr/bin/env bash
DATADIR=../../data/
N=10000 #2385643
#M = number of new links to introduce
#SR = Spread reduction (multiplied with infection probability) - this should be introduced on the day of intervention
NINITIAL=1 #How many nodes to pick in the initial step
PC=1 #Pseudo count
OUTDIR=../../results/New_York/n10000/
for M in 1 2 3 4 5 #
  do  #
    for SEED in 0 1 2 #3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99
    do
      for SR in '2_2_2_2' '4_4_4_4' '1_1_2_2' '1_1_4_4' '2_2_1_1' '4_4_1_1' '1_1_1_1'
        do
          ./model.py --datadir $DATADIR --n $N --m $M --s $SR --num_initial $NINITIAL --pseudo_count $PC --seed $SEED --outdir $OUTDIR
        done
    done
  done
