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
from scipy.stats import pearsonr, sem
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

    fig, ax = plt.subplots(figsize=(4.5/2.54, 4/2.54))
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


def plot_deaths(all_results, age_groups, num_days, observed_deaths, weeks, n, week_dates, colors, labels, outdir):
    '''Plot the deaths per age group with different links (m)
    and reductions in inf_prob
    '''

    ms = all_results['m'].unique()
    seeds = all_results['seed'].unique()

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
        total = np.zeros((len(colors.keys()),len(seeds),int(num_days/7)))
        for ag in age_groups:
            fig, ax = plt.subplots(figsize=(4.5/2.54, 4.5/2.54))
            #Go through all combos
            ti=0
            for c in colors:
                m_combo_results = m_results[m_results['combo']==c]

                #Save deaths
                ag_deaths = np.zeros((len(seeds),num_days))
                for seed in seeds:
                    m_combo_seed_results = m_combo_results[m_combo_results['seed']==seed]
                    try:
                        ag_deaths[seed,:] = np.array(m_combo_seed_results[ag+' deaths']) #Get deaths for combo and ag
                    except:
                        print(seed)
                        break
                #Scale to Stockholm
                ag_deaths = ag_deaths*(2385643/n)
                #Average
                ag_deaths_av = np.average(ag_deaths,axis=0)
                ag_deaths_std = np.std(ag_deaths,axis=0)
                x=np.arange(ag_deaths.shape[1])
                #Sum per week
                weekly_deaths = np.zeros((len(seeds),int(num_days/7)))
                weekly_deaths_av = np.zeros(int(num_days/7))
                weekly_deaths_std = np.zeros(int(num_days/7))
                for w in range(int(num_days/7)):
                    weekly_deaths[:,w]=np.sum(ag_deaths[:,w*7:(w*7)+7],axis=1)
                    weekly_deaths_av[w]=np.sum(ag_deaths_av[w*7:(w*7)+7])
                    weekly_deaths_std[w]=np.sum(ag_deaths_std[w*7:(w*7)+7])

                #Cumulative
                weekly_deaths_av = np.cumsum(weekly_deaths_av)
                weekly_deaths_std = np.cumsum(weekly_deaths_std)
                #The two first weeks for Stockholm are not considered part of the epidemic (start modeling on week 8)
                #I make sure the curves are in phase, since the phase is dependent on the initial spread, which is unknown.
                ax.plot(np.arange(5,weekly_deaths_av.shape[0]), weekly_deaths_av[:-5], color = colors[c], linewidth=1)
                ax.plot(np.arange(5,weekly_deaths_av.shape[0]), weekly_deaths_av[:-5]-weekly_deaths_std[:-5], color = colors[c], linewidth=0.5, linestyle='dashed')
                ax.plot(np.arange(5,weekly_deaths_av.shape[0]), weekly_deaths_av[:-5]+weekly_deaths_std[:-5], color = colors[c], linewidth=0.5, linestyle='dashed')
                #Add to total
                total[ti,:,:] += weekly_deaths
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
        fig, ax = plt.subplots(figsize=(4.5/2.54, 4/2.54))
        fig2, ax2 = plt.subplots(figsize=(4.5/2.54, 4/2.54))
        ti=0
        o_deaths = np.cumsum(observed_deaths)
        print(m)
        for c in colors:
            m_deaths_av = np.cumsum(np.average(total[ti,:,:],axis=0))
            m_deaths_std = np.cumsum(np.std(total[ti,:,:],axis=0))
            if c != '1_1_1_1_1_1':
                ax.plot(np.arange(total.shape[2]), m_deaths_av, color = colors[c], linewidth=1)
                ax.fill_between(np.arange(total.shape[2]),m_deaths_av-m_deaths_std,m_deaths_av+m_deaths_std,color = colors[c],alpha=0.5)
                #ax.plot(np.arange(total.shape[2]),m_deaths_av+m_deaths_std,color = colors[c],linewidth=0.5, linestyle='dashed')
            else:
                ax2.plot(np.arange(total.shape[2]), m_deaths_av, color = colors[c], linewidth=1)
                ax2.fill_between(np.arange(total.shape[2]),m_deaths_av-m_deaths_std,m_deaths_av+m_deaths_std,color = colors[c],alpha=0.5)            

            R,p = pearsonr(o_deaths,m_deaths_av)
            print(labels[c]+','+str(np.average(np.absolute(o_deaths[5:]-m_deaths_av[:-5])))+','+str(R))
            ti+=1

        ax.bar(np.arange(total.shape[2]), o_deaths, alpha = 0.5, label = 'Observation')
        plt.xticks(x_weeks, weeks, rotation='vertical')
        ax.set_title('m='+str(m))
        #ax.set_ylim(yscale[m])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('Deaths')
        fig.tight_layout()
        fig.savefig(outdir+'deaths_'+str(m)+'_total.png', format='png', dpi=300)
        plt.close()

        ax2.bar(np.arange(total.shape[2]), o_deaths, alpha = 0.5, label = 'Observation')
        plt.xticks(x_weeks, weeks, rotation='vertical')
        ax2.set_title('m='+str(m))
        #ax.set_ylim(yscale[m])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_ylabel('Deaths')
        fig2.tight_layout()
        fig2.savefig(outdir+'deaths_'+str(m)+'_total_100.png', format='png', dpi=300)
        plt.close()

    return None

def plot_cases(all_results, age_groups, num_days, n, colors, labels, outdir):
    '''Plot the cases per age group with different links (m)
    and reductions in inf_prob
    '''

    ms = all_results['m'].unique()
    seeds = all_results['seed'].unique()

    #Go through all ms
    for m in ms:
        m_results = all_results[all_results['m']==m]
        #Go through all age_groups
        total = np.zeros((len(colors.keys()),len(seeds),int(num_days)))
        for ag in age_groups:
            fig, ax = plt.subplots(figsize=(4.5/2.54, 4/2.54))
            #Go through all combos
            ti=0
            for c in colors:
                m_combo_results = m_results[m_results['combo']==c]
                #cases
                ag_cases = np.zeros((len(seeds),num_days))
                for seed in seeds:
                    m_combo_seed_results = m_combo_results[m_combo_results['seed']==seed]
                    ag_cases[seed,:] = np.array(m_combo_seed_results[ag+' cases']) #Get cases for combo and ag

                ag_cases_av = 100*np.cumsum(np.average(ag_cases,axis=0))/n
                ag_cases_std = 100*np.cumsum(np.average(ag_cases,axis=0))/n
                ax.plot(np.arange(ag_cases.shape[1]),ag_cases_av, color = colors[c], linewidth=1)
                ax.plot(np.arange(ag_cases.shape[1]),ag_cases_av-ag_cases_std,color = colors[c],linewidth=0.5, linestyle='dashed')
                ax.plot(np.arange(ag_cases.shape[1]),ag_cases_av+ag_cases_std,color = colors[c],linewidth=0.5, linestyle='dashed')
                #Add to total
                total[ti,:,:] += ag_cases
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
        fig, ax = plt.subplots(figsize=(4.5/2.54, 4/2.54))
        ti=0
        for c in colors:
            m_cases_av = 100*np.cumsum(np.average(total[ti,:,:],axis=0))/n
            m_cases_std = 100*np.cumsum(np.std(total[ti,:,:],axis=0))/n
            #plot
            ax.plot(np.arange(total.shape[2]), m_cases_av, color = colors[c], linewidth=1)
            ax.plot(np.arange(total.shape[2]),m_cases_av-m_cases_std,color = colors[c],linewidth=0.5, linestyle='dashed')
            ax.plot(np.arange(total.shape[2]),m_cases_av+m_cases_std,color = colors[c],linewidth=0.5, linestyle='dashed')
            ti+=1

        #plt.xlim([0,30])
        ax.set_title('m='+str(m))
        #ax.set_ylim(yscale[m])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Day')
        ax.set_ylabel('% Cases')
        fig.tight_layout()
        fig.savefig(outdir+'cases_'+str(m)+'_total.png', format='png', dpi=300)
        plt.close()

    return None

def plot_edges(all_results, age_groups, num_days, n, colors, labels, outdir):
    '''Plot the edges (add per age group) with different links (m)
    and reductions in inf_prob
    '''

    ms = all_results['m'].unique()
    seeds = all_results['seed'].unique()
    #Go through all ms
    for m in ms:
        m_results = all_results[all_results['m']==m]
        #Total
        fig, ax = plt.subplots(figsize=(4.5/2.54, 4/2.54))

        for c in colors:
            m_combo_results = m_results[m_results['combo']==c]
            edges = np.zeros((len(seeds),num_days))
            for seed in seeds:
                m_combo_seed_results = m_combo_results[m_combo_results['seed']==seed]
                edges[seed,:] = np.array(m_combo_seed_results['edges']) #Get cases for combo and ag
            #Average over seeds
            edges = np.average(edges,axis=0)
            ax.plot(np.arange(len(edges)),edges/max(edges), color = colors[c], linewidth=1)

        ax.set_title('m='+str(m))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Day')
        ax.set_ylabel('% Edges')
        fig.tight_layout()
        fig.savefig(outdir+'edges_'+str(m)+'_total.png', format='png', dpi=300)
        plt.close()

    return None

def plot_degrees(all_results, age_groups, num_days, n, colors, labels, outdir):
    '''Plot the max degree removed each day divided by the maximum degree in the net
    '''

    ms = all_results['m'].unique()
    seeds = all_results['seed'].unique()

    #Go through all ms
    for m in ms:
        m_results = all_results[all_results['m']==m]
        #Total
        fig, ax = plt.subplots(figsize=(4.5/2.54, 4/2.54))
        combo=1
        fetched_y = []
        for c in colors:
            m_combo_results = m_results[m_results['combo']==c]
            degrees = np.zeros((len(seeds),num_days))
            for seed in seeds:
                m_combo_seed_results = m_combo_results[m_combo_results['seed']==seed]
                degrees[seed,:] = np.array(m_combo_seed_results['perc_above_left']) #Get cases for combo and ag
            #Average over seeds
            degrees_av = 100*np.average(degrees,axis=0)
            degrees_std = 100*np.std(degrees,axis=0)
            ax.plot(np.arange(len(degrees_av)),degrees_av, color = colors[c], linewidth=1)
            ax.plot(np.arange(len(degrees_av)),degrees_av-degrees_std, color = colors[c],linewidth=0.5, linestyle='dashed')
            ax.plot(np.arange(len(degrees_av)),degrees_av+degrees_std, color = colors[c],linewidth=0.5, linestyle='dashed')

        ax.set_xlim([0,25])
        ax.set_title('m='+str(m))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Day')
        ax.set_ylabel('Nodes left above t (%)')
        fig.tight_layout()
        fig.savefig(outdir+'deg_'+str(m)+'_total.png', format='png', dpi=300)
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

#Get the results
try:
    all_results = pd.read_csv('/home/pbryant/results/COVID19/epidemic_net/Stockholm/all_results.csv')
    num_days = 210
except:
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
        resultdf['seed']=int(info.split('_')[2])
        resultdf['combo']=info[-11:]
        #append df
        all_results = all_results.append(resultdf)

    #save
    all_results.to_csv('/home/pbryant/results/COVID19/epidemic_net/Stockholm/all_results.csv')

#xticks
week_dates = ['Feb 10', 'Feb 17', 'Feb 24', 'Mar 2', 'Mar 9', 'Mar 16', 'Mar 23', 'Mar 30', 'Apr 6',
            'Apr 13', 'Apr 20', 'Apr 27', 'May 4', 'May 11', 'May 18', 'May 25', 'Jun 1', 'Jun 8', 'Jun 15',
            'Jun 22', 'Jun 29', 'Jul 6', 'Jul 13', 'Jul 20', 'Jul 27', 'Aug 3', 'Aug 10', 'Aug 17', 'Aug 24', 'Aug 31']

colors = {'1_1_1_1_1_1':'k', '2_2_2_2_2_2':'cornflowerblue', '4_4_4_4_4_4':'royalblue',
        '1_1_2_2_2_2': 'springgreen', '1_1_4_4_4_4':'mediumseagreen', '2_2_1_1_1_1':'magenta', '4_4_1_1_1_1':'darkmagenta'}
labels = {'1_1_1_1_1_1':'0-49: 100%,50+: 100%', '2_2_2_2_2_2':'0-49: 50%,50+: 50%', '4_4_4_4_4_4':'0-49: 25%,50+: 25%',
        '1_1_2_2_2_2': '0-49: 100%,50+: 50%', '1_1_4_4_4_4':'0-49: 100%,50+: 25%', '2_2_1_1_1_1':'0-49: 50%,50+: 100%',
        '4_4_1_1_1_1':'0-49: 25%,50+: 100%'}
#Plot deaths
plot_deaths(all_results, age_groups, num_days, observed_deaths, weeks, n, week_dates, colors, labels, outdir+'deaths/')

#Plot cases
plot_cases(all_results, age_groups, num_days, n, colors, labels, outdir+'cases/')

#Plot the edges
plot_edges(all_results, age_groups, num_days,  n, colors, labels, outdir+'edges/')

#Plot the max degree reomved each day
plot_degrees(all_results, age_groups, num_days, n, colors, labels, outdir+'degrees/')
#Plot the number removed - the ones that have issued spread
#plot_epidemic(np.arange(num_days), 100*np.array(num_removed)/n,'Days since initial spread','% Active spreaders','Active spreaders',m, outdir+'active_spreaders_'+str(m)+suffix)
