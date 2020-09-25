#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pystan

import pdb



#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Analyze the relationship between google mobility data and deaths''')

parser.add_argument('--us_deaths', nargs=1, type= str, default=sys.stdin, help = 'Path to death data.')
parser.add_argument('--mobility_data', nargs=1, type= str, default=sys.stdin, help = 'Path to mobility data.')
parser.add_argument('--population_sizes', nargs=1, type= str, default=sys.stdin, help = 'Path to population size data.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to outdir.')

###FUNCTIONS###

def format_data(us_deaths, mobility_data, population_sizes):
    '''Format the data by
    1. Smoothing death and mobility data
    2. Merging them on date
    '''
    pdb.set_trace()
    #Get data by state
    for c in range(len(subregions)):

        #State
        region =subregions[c]
        #Get region epidemic data
        regional_deaths = us_deaths[us_deaths['Province_State']== region]
        cols = regional_deaths.columns
        #Calculate back per day - now cumulative
        deaths_per_day = []
        dates = cols[12:]
        #First deaths
        deaths_per_day.append(np.sum(regional_deaths[dates[0]]))
        for d in range(1,len(dates)):#The first 12 columns are not deaths
            deaths_per_day.append(np.sum(regional_deaths[dates[d]])-np.sum(regional_deaths[dates[d-1]]))










#####MAIN#####
args = parser.parse_args()
us_deaths = pd.read_csv(args.us_deaths[0])
mobility_data = pd.read_csv(args.mobility_data[0])
population_sizes = pd.read_csv(args.population_sizes[0])
outdir = args.outdir[0]
