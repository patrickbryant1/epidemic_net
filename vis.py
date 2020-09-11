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
parser.add_argument('--resultsdir', nargs=1, type= str, default=sys.stdin, help = 'Path to results.')
parser.add_argument('--n', nargs=1, type= int, default=sys.stdin, help = 'Num nodes in net.')
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


def plot_deaths(all_results, age_groups, num_days):
    '''Plot the deaths per age group with different links (m)
    and reductions in inf_prob
    '''
    ms = all_results['m'].unique()
    colors = ['slategray','royalblue', 'navy','lightskyblue', 'darkcyan', 'mediumseagreen', 'paleturquoise' ]
    #Go through all ms
    for m in ms:
        m_results = all_results[all_results['m']==m]
        combos = m_results['combo'].unique()
        n_combos = len(combos)
        #Go through all age_groups
        for ag in age_groups:
            fig, ax = plt.subplots(figsize=(6/2.54, 4.5/2.54))
            #Go through all combos
            ci=0
            for c in combos:
                m_combo_results = m_results[m_results['combo']==c]
                ag_deaths = np.array(m_combo_results[ag+' deaths']) #Get deaths for combo and ag

                inf_probs
                #Sum per week
                weekly_deaths = np.zeros(int(num_days/7))
                for w in range(len(weekly_deaths)):
                    weekly_deaths[w]=np.sum(ag_deaths[w*7:(w*7)+7])
                ax.plot(np.arange(weekly_deaths.shape[0]), weekly_deaths, color = colors[ci], label = age_groups[i], linewidth=2)
                ci+=1 #increase combo index
            pdb.set_trace()

    #Plot deaths
    yscale = {1:[0,500],2:[0,2000],3:[0,3000],5:[0,4000]}
    weekly_deaths = np.zeros((len(age_groups),len(stockholm_csv)))
    #Do a 7day window to get more even death predictions
    for i in range(weekly_deaths.shape[0]):
        for j in range(weekly_deaths.shape[1]):
            weekly_deaths[i,j]=np.sum(deaths[i,j*7:(j*7)+7])
    weekly_deaths=weekly_deaths*(2385643/n) #scale with diff to Stockholm population

    fig, ax = plt.subplots(figsize=(14/2.54, 9/2.54))
    colors = ['slategray','royalblue', 'navy','lightskyblue', 'darkcyan', 'mediumseagreen', 'paleturquoise' ]
    for i in range(weekly_deaths.shape[0]):
        ax.plot(np.arange(weekly_deaths.shape[1]), weekly_deaths[i,:], color = colors[i+1], label = age_groups[i], linewidth=2)
    #Total
    ax.plot(np.arange(weekly_deaths.shape[1]), np.sum(weekly_deaths,axis=0), color = colors[0], label = 'Total', linewidth=2)
    ax.bar(np.arange(weekly_deaths.shape[1]),observed_deaths, alpha = 0.5, label = 'Observation')
    ax.legend()
    plt.xticks(np.arange(weekly_deaths.shape[1]), weeks)
    ax.set_xlabel('Week')
    ax.set_ylabel('Deaths')

    title= str(m)+' links\n'+'Age 0-49, inf.prob. '+str(1/float(s[0]))+'\nAge 50+, inf.prob. '+str(1/float(s[5]))
    ax.set_title(title)
    ax.set_ylim(yscale[m])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir+'deaths_'+str(m)+suffix, format='png', dpi=300)
#####MAIN#####
args = parser.parse_args()
datadir = args.datadir[0]
resultsdir= args.resultsdir[0]
n = args.n[0]
outdir = args.outdir[0]

#Get stockholm csv
stockholm_csv = pd.read_csv(datadir+'stockholm.csv')
observed_deaths = stockholm_csv['Antal_avlidna_vecka']
weeks = stockholm_csv['veckonummer']
#Age groups
age_groups = ['0-49','50-59','60-69','70-79','80-89','90+']
#Results
result_dfs = glob.glob(resultsdir+'*.csv')
#Loop through all results dfs
all_results = pd.DataFrame()
combos = {'1_1_1_1_1_1':1, '2_2_2_2_2_2':2, '4_4_4_4_4_4':3, '1_1_2_2_2_2':4, '1_1_4_4_4_4':5, '2_2_1_1_1_1':6, '4_4_1_1_1_1':7}

for name in result_dfs:
    resultdf = pd.read_csv(name)
    num_days = len(resultdf)
    info = name.split('/')[-1].split('.')[0].split('_')
    pdb.set_trace()
    m = int(info[1])
    resultdf['m']=m
    resultdf['combo']=combo
    combo+=1
    for a in range(len(age_groups)):
        resultdf['inf. prob. '+age_groups[a]]=np.round(1/int(info[a+2]),2)
    #append df
    all_results = all_results.append(resultdf)

#Plot deaths
plot_deaths(all_results, age_groups, num_days)

num_new_infections = np.array(resultdf['num_new_infections'])
#Get deaths

deaths = np.zeros((len(age_groups),num_days)) #6 age groups
for i in range(len(age_groups)):
    deaths[i,:]=np.array(resultdf[age_groups[i]+' deaths'])
#Get cases
cases = np.zeros((len(age_groups),num_days)) #6 age groups
for i in range(len(age_groups)):
    cases[i,:]=np.array(resultdf[age_groups[i]+' cases'])
#Removed and edges
num_removed = np.array(resultdf['num_new_removed'])
edges = np.array(resultdf['edges'])

#Suffix
suffix = ''
for si in s:
    suffix+='_'+str(si)
suffix+='.png'
#Plot spread
plot_epidemic(np.arange(num_days), 100*(num_new_infections/n),'Days since initial spread','% Infected per day', 'Daily cases', m,outdir+'cases_'+str(m)+suffix)
plot_epidemic(np.arange(num_days), 100*(np.cumsum(num_new_infections)/n),'Days since initial spread','Cumulative % infected','Cumulative cases', m,outdir+'cumulative_cases_'+str(m)+suffix)
plot_epidemic(np.arange(num_days),edges,'Days since initial spread','Remaining edges','Edges',m, outdir+'edges_'+str(m)+suffix)

#Plot cases per age group
weekly_cases = np.zeros((len(age_groups),len(stockholm_csv)))
#Do a 7day window to get more even case predictions
for i in range(weekly_cases.shape[0]):
    for j in range(weekly_cases.shape[1]):
        weekly_cases[i,j]=np.sum(cases[i,j*7:(j*7)+7])
weekly_cases=weekly_cases*(2385643/n) #scale with diff to Stockholm population

fig, ax = plt.subplots(figsize=(14/2.54, 9/2.54))
colors = ['slategray','royalblue', 'navy','lightskyblue', 'darkcyan', 'mediumseagreen', 'paleturquoise' ]
for i in range(weekly_cases.shape[0]):
    ax.plot(np.arange(weekly_cases.shape[1]), weekly_cases[i,:], color = colors[i+1], label = age_groups[i], linewidth=2)
#Total
ax.plot(np.arange(weekly_cases.shape[1]), np.sum(weekly_cases,axis=0), color = colors[0], label = 'Total', linewidth=2)
ax.legend()
plt.xticks(np.arange(weekly_cases.shape[1]), weeks)
ax.set_xlabel('Week')
ax.set_ylabel('Cases')
ax.set_title(str(m)+' links')
#ax.set_ylim([0,4000])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(outdir+'weekly_cases_'+str(m)+suffix, format='png', dpi=300)


#Plot deaths
yscale = {1:[0,500],2:[0,2000],3:[0,3000],5:[0,4000]}
weekly_deaths = np.zeros((len(age_groups),len(stockholm_csv)))
#Do a 7day window to get more even death predictions
for i in range(weekly_deaths.shape[0]):
    for j in range(weekly_deaths.shape[1]):
        weekly_deaths[i,j]=np.sum(deaths[i,j*7:(j*7)+7])
weekly_deaths=weekly_deaths*(2385643/n) #scale with diff to Stockholm population

fig, ax = plt.subplots(figsize=(14/2.54, 9/2.54))
colors = ['slategray','royalblue', 'navy','lightskyblue', 'darkcyan', 'mediumseagreen', 'paleturquoise' ]
for i in range(weekly_deaths.shape[0]):
    ax.plot(np.arange(weekly_deaths.shape[1]), weekly_deaths[i,:], color = colors[i+1], label = age_groups[i], linewidth=2)
#Total
ax.plot(np.arange(weekly_deaths.shape[1]), np.sum(weekly_deaths,axis=0), color = colors[0], label = 'Total', linewidth=2)
ax.bar(np.arange(weekly_deaths.shape[1]),observed_deaths, alpha = 0.5, label = 'Observation')
ax.legend()
plt.xticks(np.arange(weekly_deaths.shape[1]), weeks)
ax.set_xlabel('Week')
ax.set_ylabel('Deaths')

title= str(m)+' links\n'+'Age 0-49, inf.prob. '+str(1/float(s[0]))+'\nAge 50+, inf.prob. '+str(1/float(s[5]))
ax.set_title(title)
ax.set_ylim(yscale[m])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(outdir+'deaths_'+str(m)+suffix, format='png', dpi=300)

#Plot the number removed - the ones that have issued spread
plot_epidemic(np.arange(num_days), 100*np.array(num_removed)/n,'Days since initial spread','% Active spreaders','Active spreaders',m, outdir+'active_spreaders_'+str(m)+suffix)
