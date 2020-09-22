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
from scipy.stats import pearsonr,sem
import pdb



#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Simulate the epidemic development of New York on a graph network''')

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


def plot_deaths(all_results, age_groups, num_days, observed_deaths, n, x_dates, dates, colors, labels, outdir):
    '''Plot the deaths per age group with different links (m)
    and reductions in inf_prob
    '''

    ms = all_results['m'].unique()
    net_seeds = all_results['net_seed'].unique()
    np_seeds = all_results['np_seed'].unique()
    combos = all_results['combo'].unique()
    #Plot Markers
    fig, ax = plt.subplots(figsize=(3.5/2.54, 3/2.54))
    i=4
    for c in colors:
        if c=='1_1_1_1':
            continue
        ax.plot([1,1.8],[i]*2, color = colors[c], linewidth=4)
        ax.text(2.001,i,labels[c])
        i-=1
    ax.set_xlim([0.999,3.9])
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(outdir+'markers.png', format='png', dpi=300)
    plt.close()

    #Plot Markers
    fig, ax = plt.subplots(figsize=(3.5/2.54, 3/2.54))
    ax.plot([1,1.8],[i]*2, color = colors['1_1_1_1'], linewidth=4)
    ax.text(2.001,i,labels['1_1_1_1'])
    ax.set_xlim([0.999,3.9])
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(outdir+'markers_100.png', format='png', dpi=300)
    plt.close()


    #Go through all ms
    for m in ms:

        m_results = all_results[all_results['m']==m]
        #Go through all age_groups
        total = np.zeros((len(m_results['combo'].unique()),len(net_seeds)*len(np_seeds),int(num_days)))
        for ag in age_groups:
            fig, ax = plt.subplots(figsize=(4.5/2.54, 4.5/2.54))
            #Go through all combos
            ti=0
            for c in combos:
                m_combo_results = m_results[m_results['combo']==c]
                #Save deaths
                ag_deaths = np.zeros((len(net_seeds)*len(np_seeds),num_days))
                pos = 0
                for net_seed in net_seeds:
                    m_combo_net_results = m_combo_results[m_combo_results['net_seed']==net_seed]
                    for np_seed in np_seeds:
                        m_combo_np_net_results = m_combo_net_results[m_combo_net_results['np_seed']==np_seed]
                        try:
                            ag_deaths[pos,:] = np.array(m_combo_np_net_results[ag+' deaths']) #Get deaths for combo and ag
                        except:
                            pdb.set_trace()
                        pos+=1

                #Scale to New York
                ag_deaths = ag_deaths*(8336817/n)
                #Cumulative
                #ag_deaths = np.cumsum(ag_deaths,axis=1)
                #Average
                ag_deaths_av = np.average(ag_deaths,axis=0)
                ag_deaths_std = sem(ag_deaths,axis=0)
                x=np.arange(ag_deaths.shape[1])
                ax.plot(np.arange(ag_deaths_av.shape[0]),ag_deaths_av, color = colors[c], linewidth=1)
                ax.plot(np.arange(ag_deaths_av.shape[0]), ag_deaths_av-ag_deaths_std, color = colors[c], linewidth=0.5, linestyle='dashed')
                ax.plot(np.arange(ag_deaths_av.shape[0]), ag_deaths_av+ag_deaths_std, color = colors[c], linewidth=0.5, linestyle='dashed')
                #Add to total
                total[ti,:,:] +=ag_deaths
                ti+=1

            #Format and save fig
            plt.xticks(x_dates, dates, rotation='vertical')
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
        fig1, ax1 = plt.subplots(figsize=(4.5/2.54, 4/2.54))
        fig2, ax2 = plt.subplots(figsize=(4.5/2.54, 4/2.54))
        ti=0
        o_deaths = observed_deaths
        print(m)
        ax1.bar(np.arange(total.shape[2]), o_deaths, alpha = 0.5, label = 'Observation')
        ax2.bar(np.arange(total.shape[2]), o_deaths, alpha = 0.5, label = 'Observation')

        for c in combos:
            m_deaths_av = np.average(total[ti,:,:],axis=0)
            m_deaths_std = sem(total[ti,:,:],axis=0)

            if c != '1_1_1_1':
                ax1.plot(np.arange(total.shape[2]), m_deaths_av, color = colors[c], linewidth=1)
                ax1.fill_between(np.arange(total.shape[2]),m_deaths_av-m_deaths_std,m_deaths_av+m_deaths_std,color = colors[c],alpha=0.5)
                #ax1.plot(np.arange(total.shape[2]),m_deaths_av+m_deaths_std,color = colors[c],linewidth=0.5, linestyle='dashed')
            else:
                ax2.plot(np.arange(total.shape[2]), m_deaths_av, color = colors[c], linewidth=1)
                ax2.fill_between(np.arange(total.shape[2]),m_deaths_av-m_deaths_std,m_deaths_av+m_deaths_std,color = colors[c],alpha=0.5)
                #ax2.plot(np.arange(total.shape[2]),m_deaths_av+m_deaths_std,color = colors[c],linewidth=0.5, linestyle='dashed')

            R,p = pearsonr(o_deaths,m_deaths_av)
            print(labels[c]+','+str(np.average(np.absolute(o_deaths-m_deaths_av)))+','+str(R))
            ti+=1


        ax1.set_xticks(x_dates)
        ax1.set_xticklabels(dates, rotation='vertical')
        ax1.set_title('m='+str(m))
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_ylabel('Deaths')
        fig1.tight_layout()
        fig1.savefig(outdir+'deaths_'+str(m)+'_total.png', format='png', dpi=300)

        ax2.set_title('m='+str(m))
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_ylabel('Deaths')
        fig2.tight_layout()
        fig2.savefig(outdir+'deaths_'+str(m)+'_total_100.png', format='png', dpi=300)

        plt.close()

        #Investigate the relationship btw randomness and final result
        i=0
        np_seed_results = {}
        total = np.cumsum(total,axis=2)
        for net_seed in net_seeds:
            for np_seed in np_seeds:
                if np_seed in [*np_seed_results.keys()]:
                    np_seed_results[np_seed].append(total[:,i,-1])
                else:
                    np_seed_results[np_seed] = [total[:,i,-1]]
                i+=1

        #Plot
        fig, ax = plt.subplots(figsize=(4.5/2.54, 4/2.54))
        for key in np_seed_results:
            av_seed_res = np.average(np.array(np_seed_results[key]))

            sns.distplot(np.array(np_seed_results[key]))
        fig.tight_layout()
        fig.savefig(outdir+'deaths_'+str(m)+'_np_seed.png', format='png', dpi=300)
        plt.close()

    return None

def plot_cases(all_results, age_groups, num_days, n, colors, labels, outdir):
    '''Plot the cases per age group with different links (m)
    and reductions in inf_prob
    '''

    ms = all_results['m'].unique()
    net_seeds = all_results['net_seed'].unique()
    np_seeds = all_results['np_seed'].unique()
    combos = all_results['combo'].unique()
    #Go through all ms
    for m in ms:
        m_results = all_results[all_results['m']==m]
        #Go through all age_groups
        total = np.zeros((len(colors.keys()),len(net_seeds)*len(np_seeds),int(num_days)))
        for ag in age_groups:
            fig, ax = plt.subplots(figsize=(4.5/2.54, 4/2.54))
            #Go through all combos
            ti=0
            for c in combos:
                m_combo_results = m_results[m_results['combo']==c]
                #cases
                ag_cases = np.zeros((len(net_seeds)*len(np_seeds),num_days))
                pos = 0
                for net_seed in net_seeds:
                    m_combo_net_results = m_combo_results[m_combo_results['net_seed']==net_seed]
                    for np_seed in np_seeds:
                        m_combo_np_net_results = m_combo_net_results[m_combo_net_results['np_seed']==np_seed]
                        try:
                            ag_cases[pos,:] = np.array(m_combo_np_net_results[ag+' cases']) #Get deaths for combo and ag
                        except:
                            pdb.set_trace()
                        pos+=1

                ag_cases_av = 100*np.cumsum(np.average(ag_cases,axis=0))/n
                ag_cases_std = 100*np.cumsum(sem(ag_cases,axis=0))/n
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
        fig1, ax1 = plt.subplots(figsize=(4.5/2.54, 4/2.54))
        fig2, ax2 = plt.subplots(figsize=(4.5/2.54, 4/2.54))
        ti=0
        for c in combos:
            m_cases_av = 100*(np.average(total[ti,:,:],axis=0))/n
            m_cases_std = 100*(sem(total[ti,:,:],axis=0))/n
            if c != '1_1_1_1':
                #plot
                ax1.plot(np.arange(total.shape[2]), m_cases_av, color = colors[c], linewidth=1)
                ax1.fill_between(np.arange(total.shape[2]),m_cases_av-m_cases_std,m_cases_av+m_cases_std,color = colors[c],alpha=0.5)
                #ax1.plot(np.arange(total.shape[2]),m_cases_av+m_cases_std,color = colors[c],linewidth=0.5, linestyle='dashed')
            else:
                ax2.plot(np.arange(total.shape[2]), m_cases_av, color = colors[c], linewidth=1)
                ax2.plot(np.arange(total.shape[2]),m_cases_av-m_cases_std,color = colors[c],linewidth=0.5, linestyle='dashed')
                ax2.plot(np.arange(total.shape[2]),m_cases_av+m_cases_std,color = colors[c],linewidth=0.5, linestyle='dashed')
            ti+=1

        ax1.set_xticks(x_dates)
        ax1.set_xticklabels(dates, rotation='vertical')
        ax1.set_title('m='+str(m))
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xlabel('Day')
        ax1.set_ylabel('% Cases')
        fig1.tight_layout()
        fig1.savefig(outdir+'cases_'+str(m)+'_total.png', format='png', dpi=300)

        ax2.set_title('m='+str(m))
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xlabel('Day')
        ax2.set_ylabel('% Cases')
        fig2.tight_layout()
        fig2.savefig(outdir+'cases_'+str(m)+'_total_100.png', format='png', dpi=300)
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
        #edges
        edges = np.zeros((len(seeds),num_days))
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

        fig1, ax1 = plt.subplots(figsize=(4.5/2.54, 4/2.54))
        fig2, ax2 = plt.subplots(figsize=(4.5/2.54, 4/2.54))
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
            if c != '1_1_1_1':
                ax1.plot(np.arange(len(degrees_av)),degrees_av, color = colors[c], linewidth=1)
                ax1.plot(np.arange(len(degrees_av)),degrees_av-degrees_std, color = colors[c],linewidth=0.5, linestyle='dashed')
                ax1.plot(np.arange(len(degrees_av)),degrees_av+degrees_std, color = colors[c],linewidth=0.5, linestyle='dashed')
            else:
                ax2.plot(np.arange(len(degrees_av)),degrees_av, color = colors[c], linewidth=1)
                ax2.plot(np.arange(len(degrees_av)),degrees_av-degrees_std, color = colors[c],linewidth=0.5, linestyle='dashed')
                ax2.plot(np.arange(len(degrees_av)),degrees_av+degrees_std, color = colors[c],linewidth=0.5, linestyle='dashed')
        ax1.set_xlim([0,25])
        ax1.set_title('m='+str(m))
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Nodes left above t (%)')
        fig1.tight_layout()
        fig1.savefig(outdir+'deg_'+str(m)+'_total.png', format='png', dpi=300)

        ax2.set_xlim([0,25])
        ax2.set_title('m='+str(m))
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Nodes left above t (%)')
        fig2.tight_layout()
        fig2.savefig(outdir+'deg_'+str(m)+'_total.png', format='png', dpi=300)
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

#Get epidemic data
epidemic_data = pd.read_csv(datadir+'new_york.csv')
observed_deaths = epidemic_data['Deaths']

#Age groups
age_groups = ['0-19','20-49','50-69','70+']
#Get the results
try:
    all_results = pd.read_csv('/home/pbryant/results/COVID19/epidemic_net/New_York/all_results.csv')
    num_days = 198

except:

    result_dfs = glob.glob(resultsdir+'*.csv')
    #Loop through all results dfs
    all_results = pd.DataFrame()
    combos = {'1_1_1_1':1, '2_2_2_2':2, '4_4_4_4':3, '1_1_2_2':4, '1_1_4_4':5, '2_2_1_1':6, '4_4_1_1':7}

    for name in result_dfs:
        resultdf = pd.read_csv(name)
        num_days = len(resultdf)
        info = name.split('/')[-1].split('.')[0]
        m = int(info.split('_')[1])
        resultdf['m']=m
        resultdf['net_seed']=int(info.split('_')[2])
        resultdf['np_seed']=int(info.split('_')[3])
        resultdf['combo']=info[-7:]

        #append df
        all_results = all_results.append(resultdf)
    #save
    all_results.to_csv('/home/pbryant/results/COVID19/epidemic_net/New_York/all_results.csv')

#xticks
x_dates = [  0,  28,  56,  84, 112, 140, 168, 197]
dates = ['Feb 29', 'Mar 28', 'Apr 25','May 23', 'Jun 20','Jul 18','Aug 15', 'Sep 13']
colors = {'1_1_1_1':'k', '2_2_2_2':'cornflowerblue', '4_4_4_4':'royalblue',
        '1_1_2_2': 'springgreen', '1_1_4_4':'mediumseagreen', '2_2_1_1':'magenta', '4_4_1_1':'darkmagenta'}
labels = {'1_1_1_1':'0-49: 100%,50+: 100%', '2_2_2_2':'0-49: 50%,50+: 50%', '4_4_4_4':'0-49: 25%,50+: 25%',
        '1_1_2_2': '0-49: 100%,50+: 50%', '1_1_4_4':'0-49: 100%,50+: 25%', '2_2_1_1':'0-49: 50%,50+: 100%',
        '4_4_1_1':'0-49: 25%,50+: 100%'}
#Plot deaths
plot_deaths(all_results, age_groups, num_days, observed_deaths, n, x_dates, dates, colors, labels, outdir+'deaths/')

#Plot cases
plot_cases(all_results, age_groups, num_days, n, colors, labels, outdir+'cases/')

#Plot the edges
#plot_edges(all_results, age_groups, num_days,  n, colors, labels, outdir+'edges/')

#Plot the max degree reomved each day
#plot_degrees(all_results, age_groups, num_days, n, colors, labels, outdir+'degrees/')
#Plot the number removed - the ones that have issued spread
#plot_epidemic(np.arange(num_days), 100*np.array(num_removed)/n,'Days since initial spread','% Active spreaders','Active spreaders',m, outdir+'active_spreaders_'+str(m)+suffix)
