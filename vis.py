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
    ax.plot(x,y, color = 'cornflowerblue')
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
observed_deaths = stockholm_csv['Antal_avlidna_vecka']
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
weekly_deaths = np.zeros(len(stockholm_csv))
#Do a 7day window to get more even death predictions
for i in range(len(weekly_deaths)):
    weekly_deaths[i]=np.sum(deaths[i*7:(i*7)+7])

fig, ax = plt.subplots(figsize=(6/2.54, 4/2.54))
ax.plot(np.arange(len(weekly_deaths)), weekly_deaths, color = 'cornflowerblue', label = 'Simulation')
ax.bar(np.arange(len(weekly_deaths)),observed_deaths, alpha = 0.5, label = 'Observation')
ax.legend()
ax.set_xlabel('Weeks since initial spread')
ax.set_ylabel('Deaths')
ax.set_title('Weekly deaths')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(outdir+'deaths.png', format='png', dpi=300)

#Plot the number removed - the ones that have issued spread
plot_epidemic(np.arange(num_days), 100*np.array(num_removed)/n,'Days since initial spread','% Active spreaders','Active spreaders', outdir+'active_spreaders.png')