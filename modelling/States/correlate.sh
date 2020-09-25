#!/usr/bin/env bash

DEATHS=../../data/USA_deaths.csv
MOBILITY=../../data/Global_Mobility_US.csv
POPSIZES=../../data/us_state_population.csv
OUTDIR=../../results/USA/
./correlate_death_mob.py --us_deaths $DEATHS --mobility_data $MOBILITY --population_sizes $POPSIZES --outdir $OUTDIR
