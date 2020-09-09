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
import pdb



#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Simulate the epidemic development of Stockholm on a graph network''')

parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to datadir.')
parser.add_argument('--resultdf', nargs=1, type= str, default=sys.stdin, help = 'Path to results.')
parser.add_argument('--n', nargs=1, type= int, default=sys.stdin, help = 'Num nodes in net.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to outdir.')

def plot_epidemic(x,y,xlabel,ylabel,title,outname):
    '''Plot the epidemic
    '''
    #Set font size
    matplotlib.rcParams.update({'font.size': 7})
    fig, ax = plt.subplots(figsize=(6/2.54, 4/2.54))
    ax.plot(x,y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(outname, format='png', dpi=300)
    plt.close()




#####MAIN#####
args = parser.parse_args()
datadir = args.datadir[0]
resultdf= pd.read_csv(args.resultdf[0])
n = args.n[0]
outdir = args.outdir[0]

#Get stockholm csv
stockholm_csv = pd.read_csv(datadir+'stockholm.csv')

#Results
num_days = len(resultdf)
num_new_infections = np.array(resultdf['num_new_infections'])
deaths = np.array(resultdf['deaths'])
num_removed = np.array(resultdf['num_new_removed'])
edges = np.array(resultdf['edges'])

#Plot spread
plot_epidemic(np.arange(num_days), 100*(num_new_infections/n),'Days since initial spread','% Infected per day', 'Daily cases', outdir+'cases.png')
plot_epidemic(np.arange(num_days), 100*(np.cumsum(num_new_infections)/n),'Days since initial spread','Cumulative % infected','Cumulative cases', outdir+'cumulative_cases.png')
plot_epidemic(np.arange(num_days),edges,'Days since initial spread','Remaining edges','Edges', outdir+'edges.png')


#Plot deaths
#Sliding window
plot_epidemic(np.arange(num_days), deaths,'Days since initial spread','Deaths','Daily deaths', outdir+'deaths.png')

#Plot the number removed - the ones that have issued spread
plot_epidemic(np.arange(num_days), 100*np.array(num_removed)/n,'Days since initial spread','% Active spreaders','Active spreaders', outdir+'active_spreaders.png')
