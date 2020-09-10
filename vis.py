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
parser.add_argument('--m', nargs=1, type= int, default=sys.stdin, help = 'Num links to add for each new node in the preferential attachment graph.')
parser.add_argument('--s', nargs=1, type= float, default=sys.stdin, help = 'Spread reduction. Float to multiply infection probability with.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to outdir.')

def plot_epidemic(x,y,xlabel,ylabel,title,m,outname):
    '''Plot the epidemic
    '''
    #Set font size
    matplotlib.rcParams.update({'font.size': 7})
    fig, ax = plt.subplots(figsize=(6/2.54, 4/2.54))
    ax.plot(x,y, color = 'cornflowerblue', label=str(m))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    fig.tight_layout()
    fig.savefig(outname, format='png', dpi=300)
    plt.close()




#####MAIN#####
args = parser.parse_args()
datadir = args.datadir[0]
resultdf= pd.read_csv(args.resultdf[0])
n = args.n[0]
m = args.m[0]
s = args.s[0]
outdir = args.outdir[0]

#Get stockholm csv
stockholm_csv = pd.read_csv(datadir+'stockholm.csv')
observed_deaths = stockholm_csv['Antal_avlidna_vecka']
weeks = stockholm_csv['veckonummer']
#Results
num_days = len(resultdf)
num_new_infections = np.array(resultdf['num_new_infections'])
#Get deaths
age_groups = ['0-49','50-59','60-69','70-79','80-89','90+']
deaths = np.zeros((len(age_groups),num_days)) #6 age groups
for i in range(len(age_groups)):
    deaths[i,:]=np.array(resultdf[age_groups[i]+' deaths'])

num_removed = np.array(resultdf['num_new_removed'])
edges = np.array(resultdf['edges'])

#Plot spread
plot_epidemic(np.arange(num_days), 100*(num_new_infections/n),'Days since initial spread','% Infected per day', 'Daily cases', m,outdir+'cases_'+str(m)+'_'+str(s)+'.png')
plot_epidemic(np.arange(num_days), 100*(np.cumsum(num_new_infections)/n),'Days since initial spread','Cumulative % infected','Cumulative cases', m,outdir+'cumulative_cases_'+str(m)+'_'+str(s)+'.png')
plot_epidemic(np.arange(num_days),edges,'Days since initial spread','Remaining edges','Edges',m, outdir+'edges_'+str(m)+'_'+str(s)+'.png')


#Plot deaths
weekly_deaths = np.zeros((len(age_groups),len(stockholm_csv)))
#Do a 7day window to get more even death predictions
for i in range(weekly_deaths.shape[0]):
    for j in range(weekly_deaths.shape[1]):
        weekly_deaths[i,j]=np.sum(deaths[i,j*7:(j*7)+7])
weekly_deaths=weekly_deaths*(2385643/n) #scale with diff to Stockholm population

fig, ax = plt.subplots(figsize=(14/2.54, 9/2.54))
colors = ['slategray','royalblue', 'navy','lightskyblue', 'darkcyan', 'mediumseagreen', 'paleturquoise' ]
for i in range(weekly_deaths.shape[0]):
    ax.plot(np.arange(weekly_deaths.shape[1]), weekly_deaths[i,:], color = colors[i+1], label = age_groups[i], linewidth=3)
#Total
ax.plot(np.arange(weekly_deaths.shape[1]), np.sum(weekly_deaths,axis=0), color = colors[0], label = 'Total', linewidth=3)
ax.bar(np.arange(weekly_deaths.shape[1]),observed_deaths, alpha = 0.5, label = 'Observation')
ax.legend()
plt.xticks(np.arange(weekly_deaths.shape[1]), weeks)
ax.set_xlabel('Week')
ax.set_ylabel('Deaths')
ax.set_title(str(m)+' links')
ax.set_ylim([0,4000])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(outdir+'deaths_'+str(m)+'_'+str(s)+'.png', format='png', dpi=300)

#Plot the number removed - the ones that have issued spread
plot_epidemic(np.arange(num_days), 100*np.array(num_removed)/n,'Days since initial spread','% Active spreaders','Active spreaders',m, outdir+'active_spreaders_'+str(m)+'_'+str(s)+'.png')
