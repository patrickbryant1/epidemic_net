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


def plot_deaths(all_results, age_groups, num_days, observed_deaths, weeks, n, week_dates, outdir):
    '''Plot the deaths per age group with different links (m)
    and reductions in inf_prob
    '''

    ms = all_results['m'].unique()
    colors = {'1_1_1_1_1_1':'royalblue', '2_2_2_2_2_2':'navy', '4_4_4_4_4_4':'magenta',
            '1_1_2_2_2_2': 'darkcyan', '1_1_4_4_4_4':'mediumseagreen', '2_2_1_1_1_1':'paleturquoise', '4_4_1_1_1_1':'darkkhaki'}
    labels = {'1_1_1_1_1_1':'0-49: 100%,50+: 100%', '2_2_2_2_2_2':'0-49: 50%,50+: 50%', '4_4_4_4_4_4':'0-49: 25%,50+: 25%',
            '1_1_2_2_2_2': '0-49: 100%,50+: 50%', '1_1_4_4_4_4':'0-49: 100%,50+: 25%', '2_2_1_1_1_1':'0-49: 50%,50+: 100%',
            '4_4_1_1_1_1':'0-49: 25%,50+: 100%'}
    yscale = {1:[0,500],2:[0,2000],3:[0,3000],5:[0,4000]}
    #weeks = weeks[np.arange(0,len(weeks),4)]


    x_weeks = [ 0,  4,  8, 12, 16, 20, 24, 29]
    weeks = np.array(week_dates)[x_weeks]


    #Plot Markers
    fig, ax = plt.subplots(figsize=(3.5/2.54, 3/2.54))
    i=5
    for c in colors:
        ax.plot([1,1.8],[i]*2, color = colors[c], linewidth=4)
        ax.text(2.001,i,labels[c])
        i-=1
    ax.set_xlim([0.999,3.9])
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(outdir+'markers.png', format='png', dpi=300)
    plt.close()

    #Go through all ms
    for m in ms:
        m_results = all_results[all_results['m']==m]
        #Go through all age_groups
        total = np.zeros((len(colors.keys()),int(num_days/7)))
        for ag in age_groups:
            fig, ax = plt.subplots(figsize=(4.5/2.54, 4.5/2.54))
            #Go through all combos
            ti=0
            for c in colors:
                m_combo_results = m_results[m_results['combo']==c]
                ag_deaths = np.array(m_combo_results[ag+' deaths']) #Get deaths for combo and ag
                #Sum per week
                weekly_deaths = np.zeros(int(num_days/7))
                for w in range(len(weekly_deaths)):
                    weekly_deaths[w]=np.sum(ag_deaths[w*7:(w*7)+7])
                #Scale to Stockohlm
                weekly_deaths = weekly_deaths*(2385643/n)
                #The two first weeks for Stockholm are not considered part of the epidemic (start modeling on week 8)
                #I make sure the curves are in phase, since the phase is dependent on the initial spread, which is unknown.
                ax.plot(np.arange(5,weekly_deaths.shape[0]), np.cumsum(weekly_deaths[:-5]), color = colors[c], linewidth=1)
                #Add to total
                total[ti,:] += weekly_deaths
                ti+=1
            #Format and save fig
            plt.xticks(x_weeks, weeks, rotation='vertical')
            ax.set_xlabel('Week')
            ax.set_ylabel('Deaths')

            title= 'Ages '+ag+'|m='+str(m)
            ax.set_title(title)
            #ax.set_ylim(yscale[m])
            ax.set_ylabel('Deaths')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            fig.savefig(outdir+'deaths_'+str(m)+'_'+ag+'.png', format='png', dpi=300)
            plt.close()


        #Total
        fig, ax = plt.subplots(figsize=(3.5/2.54, 3/2.54))
        ti=0
        for c in colors:
            ax.plot(np.arange(5,total.shape[1]), np.cumsum(total[ti,:-5]), color = colors[c], linewidth=1)
            ti+=1
        ax.bar(np.arange(total.shape[1]), np.cumsum(observed_deaths), alpha = 0.5, label = 'Observation')
        plt.xticks(x_weeks, weeks, rotation='vertical')
        ax.set_title('m='+str(m))
        #ax.set_ylim(yscale[m])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('Deaths')
        fig.tight_layout()
        fig.savefig(outdir+'deaths_'+str(m)+'_total.png', format='png', dpi=300)
        plt.close()

    return None

def plot_cases(all_results, age_groups, num_days, n, outdir):
    '''Plot the cases per age group with different links (m)
    and reductions in inf_prob
    '''

    ms = all_results['m'].unique()
    colors = {'1_1_1_1_1_1':'royalblue', '2_2_2_2_2_2':'navy', '4_4_4_4_4_4':'magenta',
            '1_1_2_2_2_2': 'darkcyan', '1_1_4_4_4_4':'mediumseagreen', '2_2_1_1_1_1':'paleturquoise', '4_4_1_1_1_1':'darkkhaki'}
    labels = {'1_1_1_1_1_1':'0-49: 100%,50+: 100%', '2_2_2_2_2_2':'0-49: 50%,50+: 50%', '4_4_4_4_4_4':'0-49: 25%,50+: 25%',
            '1_1_2_2_2_2': '0-49: 100%,50+: 50%', '1_1_4_4_4_4':'0-49: 100%,50+: 25%', '2_2_1_1_1_1':'0-49: 50%,50+: 100%',
            '4_4_1_1_1_1':'0-49: 25%,50+: 100%'}
    yscale = {1:[0,500],2:[0,2000],3:[0,3000],5:[0,4000]}
    #Go through all ms
    for m in ms:
        m_results = all_results[all_results['m']==m]
        #Go through all age_groups
        total = np.zeros((len(colors.keys()),int(num_days)))
        for ag in age_groups:
            fig, ax = plt.subplots(figsize=(4.5/2.54, 4.5/2.54))
            #Go through all combos
            ti=0
            for c in colors:
                m_combo_results = m_results[m_results['combo']==c]
                ag_cases = np.array(m_combo_results[ag+' cases']) #Get cases for combo and ag
                #Sum per week
                #weekly_cases = np.zeros(int(num_days/7))
                #for w in range(len(weekly_cases)):
                #    weekly_cases[w]=np.sum(ag_cases[w*7:(w*7)+7])
                #Scale to Stockohlm
                #weekly_cases = weekly_cases*(2385643/n)
                #The two first weeks for Stockholm are not considered part of the epidemic (start modeling on week 8)
                #I make sure the curves are in phase, since the phase is dependent on the initial spread, which is unknown.
                ax.plot(np.arange(ag_cases.shape[0]),100*np.cumsum(ag_cases)/n, color = colors[c], linewidth=1)
                #Add to total
                total[ti,:] += ag_cases
                ti+=1
            #Format and save fig
            ax.set_xlabel('Day')
            ax.set_ylabel('% Cases')
            title= 'Ages '+ag+'|m='+str(m)
            ax.set_title(title)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            fig.savefig(outdir+'cases_'+str(m)+'_'+ag+'.png', format='png', dpi=300)
            plt.close()


        #Total
        fig, ax = plt.subplots(figsize=(3.5/2.54, 3/2.54))
        ti=0
        for c in colors:
            ax.plot(np.arange(total.shape[1]),100*np.cumsum(total[ti,:])/n, color = colors[c], linewidth=1)
            ti+=1

        #plt.xlim([0,30])
        ax.set_title('m='+str(m))
        #ax.set_ylim(yscale[m])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('% Cases')
        fig.tight_layout()
        fig.savefig(outdir+'cases_'+str(m)+'_total.png', format='png', dpi=300)
        plt.close()

    return None

def plot_edges(all_results, age_groups, num_days, n, outdir):
    '''Plot the edges (add per age group) with different links (m)
    and reductions in inf_prob
    '''

    ms = all_results['m'].unique()
    colors = {'1_1_1_1_1_1':'royalblue', '2_2_2_2_2_2':'navy', '4_4_4_4_4_4':'magenta',
            '1_1_2_2_2_2': 'darkcyan', '1_1_4_4_4_4':'mediumseagreen', '2_2_1_1_1_1':'paleturquoise', '4_4_1_1_1_1':'darkkhaki'}
    labels = {'1_1_1_1_1_1':'0-49: 100%,50+: 100%', '2_2_2_2_2_2':'0-49: 50%,50+: 50%', '4_4_4_4_4_4':'0-49: 25%,50+: 25%',
            '1_1_2_2_2_2': '0-49: 100%,50+: 50%', '1_1_4_4_4_4':'0-49: 100%,50+: 25%', '2_2_1_1_1_1':'0-49: 50%,50+: 100%',
            '4_4_1_1_1_1':'0-49: 25%,50+: 100%'}
    yscale = {1:[0,500],2:[0,2000],3:[0,3000],5:[0,4000]}


    #Go through all ms
    for m in ms:
        m_results = all_results[all_results['m']==m]
        #Total
        fig, ax = plt.subplots(figsize=(3.5/2.54, 3/2.54))

        for c in colors:
            m_combo_results = m_results[m_results['combo']==c]
            ax.plot(np.arange(len(m_combo_results)),np.array(m_combo_results['edges'])/max(m_combo_results['edges']), color = colors[c], linewidth=1)

        ax.set_title('m='+str(m))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('% Edges')
        fig.tight_layout()
        fig.savefig(outdir+'edges_'+str(m)+'_total.png', format='png', dpi=300)
        plt.close()

    return None
#####MAIN#####
#Set font size
matplotlib.rcParams.update({'font.size': 5.5})
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
    info = name.split('/')[-1].split('.')[0]
    m = int(info.split('_')[1])
    resultdf['m']=m
    resultdf['combo']=info[-11:]
    #append df
    all_results = all_results.append(resultdf)

#xticks
week_dates = ['Feb 10', 'Feb 17', 'Feb 24', 'Mar 2', 'Mar 9', 'Mar 16', 'Mar 23', 'Mar 30', 'Apr 6',
            'Apr 13', 'Apr 20', 'Apr 27', 'May 4', 'May 11', 'May 18', 'May 25', 'Jun 1', 'Jun 8', 'Jun 15',
            'Jun 22', 'Jun 29', 'Jul 6', 'Jul 13', 'Jul 20', 'Jul 27', 'Aug 3', 'Aug 10', 'Aug 17', 'Aug 24', 'Aug 31']
#Plot deaths
plot_deaths(all_results, age_groups, num_days, observed_deaths, weeks, n, week_dates, outdir+'deaths/')

#Plot cases
plot_cases(all_results, age_groups, num_days, n, outdir+'cases/')

#Plot the edges
plot_edges(all_results, age_groups, num_days,  n, outdir+'edges/')
#Plot the number removed - the ones that have issued spread
#plot_epidemic(np.arange(num_days), 100*np.array(num_removed)/n,'Days since initial spread','% Active spreaders','Active spreaders',m, outdir+'active_spreaders_'+str(m)+suffix)
