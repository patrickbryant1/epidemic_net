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
    alphas = all_results['alpha'].unique()

    #Plot Markers
    fig, ax = plt.subplots(figsize=(3.5/2.54, 3/2.54))
    i=4
    for c in colors:
        ax.plot([1,1.8],[i]*2, color = colors[c], linewidth=4)
        ax.text(2.001,i,'alpha = '+c)
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
        total = np.zeros((len(combos),len(alphas),len(net_seeds)*len(np_seeds),int(num_days)))
        for ag in age_groups:
            fig, ax = plt.subplots(figsize=(4.5/2.54, 4.5/2.54))
            #Go through all combos
            ci = -1 #combo index
            for c in combos:
                m_combo_results = m_results[m_results['combo']==c]
                ci+=1
                ai = -1 #Alpha index
                #Go through all alphas
                for alpha in alphas:
                    #Save deaths
                    ag_deaths = np.zeros((len(net_seeds)*len(np_seeds),num_days))
                    m_combo_alpha_results = m_combo_results[m_combo_results['alpha']==alpha]
                    ai+=1
                    #go through all net seeds
                    ni=-1 #net index
                    for net_seed in net_seeds:
                        m_combo_alpha_net_results = m_combo_alpha_results[m_combo_alpha_results['net_seed']==net_seed]
                        #go through all np seeds
                        for np_seed in np_seeds:
                            ni+=1
                            m_combo_alpha_net_np_results = m_combo_alpha_net_results[m_combo_alpha_net_results['np_seed']==np_seed]
                            ag_deaths[ni,:] = np.array(m_combo_alpha_net_np_results[ag+' deaths']) #Get deaths for combo and ag

                    #Scale to New York
                    ag_deaths = ag_deaths*(47329979/n)

                    #Cumulative
                    #ag_deaths = np.cumsum(ag_deaths,axis=1)
                    #Average
                    ag_deaths_av = np.average(ag_deaths,axis=0)
                    ag_deaths_std = sem(ag_deaths,axis=0)
                    x=np.arange(ag_deaths.shape[1])
                    ax.plot(np.arange(ag_deaths_av.shape[0]),ag_deaths_av, color = colors[str(alpha)], linewidth=1)
                    ax.plot(np.arange(ag_deaths_av.shape[0]), ag_deaths_av-ag_deaths_std, color = colors[str(alpha)], linewidth=0.5, linestyle='dashed')
                    ax.plot(np.arange(ag_deaths_av.shape[0]), ag_deaths_av+ag_deaths_std, color =colors[str(alpha)], linewidth=0.5, linestyle='dashed')
                    #Add to total

                    total[ci,ai,:,:] +=ag_deaths


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
        ci=0
        for c in combos:
            fig1, ax1 = plt.subplots(figsize=(4.5/2.54, 4/2.54))
            o_deaths = observed_deaths
            print(m)
            ax1.bar(np.arange(len(o_deaths)), o_deaths, alpha = 0.5, label = 'Observation')
            ai=0
            for a in alphas:
                m_deaths_av = np.average(total[ci,ai,:,:],axis=0)
                m_deaths_std = sem(total[ci,ai,:,:],axis=0)
                ax1.plot(np.arange(len(m_deaths_av)), m_deaths_av, color = colors[str(a)], linewidth=1, label = str(a))
                ax1.fill_between(np.arange(len(m_deaths_av)),m_deaths_av-m_deaths_std,m_deaths_av+m_deaths_std,color = colors[str(a)],alpha=0.5)

                R,p = pearsonr(o_deaths,m_deaths_av)
                print(labels[c]+','+str(np.average(np.absolute(o_deaths-m_deaths_av)))+','+str(R))

                ai+=1
            ci+=1

            ax1.set_xticks(x_dates)
            ax1.set_xticklabels(dates, rotation='vertical')
            ax1.set_title('m='+str(m)+'|'+labels[c])
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.set_ylabel('Deaths')
            fig1.tight_layout()
            fig1.savefig(outdir+'deaths_'+str(m)+'_'+str(c)+'_total.png', format='png', dpi=300)
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

def plot_cases(all_results, age_groups, num_days, n, colors, labels, x_dates, dates, outdir):
    '''Plot the cases per age group with different links (m)
    and reductions in inf_prob
    '''

    ms = all_results['m'].unique()
    net_seeds = all_results['net_seed'].unique()
    np_seeds = all_results['np_seed'].unique()
    combos = all_results['combo'].unique()
    alphas = all_results['alpha'].unique()

    #Go through all ms
    for m in ms:

        m_results = all_results[all_results['m']==m]
        #Go through all age_groups
        total = np.zeros((len(combos),len(alphas),len(net_seeds)*len(np_seeds),int(num_days)))
        for ag in age_groups:
            fig, ax = plt.subplots(figsize=(4.5/2.54, 4.5/2.54))
            #Go through all combos
            ci = -1 #combo index
            for c in combos:
                m_combo_results = m_results[m_results['combo']==c]
                ci+=1
                ai = -1 #Alpha index
                #Go through all alphas
                for alpha in alphas:
                    #Save cases
                    ag_cases = np.zeros((len(net_seeds)*len(np_seeds),num_days))
                    m_combo_alpha_results = m_combo_results[m_combo_results['alpha']==alpha]
                    ai+=1
                    #go through all net seeds
                    ni=-1 #net index
                    for net_seed in net_seeds:
                        m_combo_alpha_net_results = m_combo_alpha_results[m_combo_alpha_results['net_seed']==net_seed]
                        #go through all np seeds
                        for np_seed in np_seeds:
                            ni+=1
                            m_combo_alpha_net_np_results = m_combo_alpha_net_results[m_combo_alpha_net_results['np_seed']==np_seed]
                            ag_cases[ni,:] = np.array(m_combo_alpha_net_np_results[ag+' cases']) #Get deaths for combo and ag


                    #Cumulative
                    ag_cases_av = 100*np.cumsum(np.average(ag_cases,axis=0))/n
                    ag_cases_std = 100*np.cumsum(sem(ag_cases,axis=0))/n


                    #Add to total
                    total[ci,ai,:,:] +=ag_cases_av



        #Total
        ci=0
        for c in combos:
            fig1, ax1 = plt.subplots(figsize=(4.5/2.54, 4/2.54))
            print(m)
            ai=0
            for a in alphas:
                m_cases_av = np.average(total[ci,ai,:,:],axis=0)
                m_cases_std = sem(total[ci,ai,:,:],axis=0)
                ax1.plot(np.arange(m_cases_av.shape[0]), m_cases_av, color = colors[str(a)], linewidth=1, label = str(a))
                ax1.fill_between(np.arange(m_cases_av.shape[0]),m_cases_av-m_cases_std,m_cases_av+m_cases_std,color = colors[str(a)],alpha=0.5)

                ai+=1
            ci+=1

            ax1.set_xticks(x_dates)
            ax1.set_xticklabels(dates, rotation='vertical')
            ax1.set_title('m='+str(m)+'|'+labels[c])
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.set_ylabel('% Cases')
            fig1.tight_layout()
            fig1.savefig(outdir+'cases_'+str(m)+'_'+str(c)+'_total.png', format='png', dpi=300)
            plt.close()

    return None

def plot_degrees(all_results, age_groups, num_days, n, colors, labels, x_dates, dates, outdir):
    '''Plot the max degree removed each day divided by the maximum degree in the net
    '''


    ms = all_results['m'].unique()
    net_seeds = all_results['net_seed'].unique()
    np_seeds = all_results['np_seed'].unique()
    combos = all_results['combo'].unique()
    alphas = all_results['alpha'].unique()

    #Go through all ms
    for m in ms:

        m_results = all_results[all_results['m']==m]
        #Go through all age_groups
        total = np.zeros((len(combos),len(alphas),len(net_seeds)*len(np_seeds),int(num_days)))


        #Go through all combos
        ci = -1 #combo index
        for c in combos:
            m_combo_results = m_results[m_results['combo']==c]
            ci+=1
            ai = -1 #Alpha index
            #Go through all alphas
            for alpha in alphas:
                #Save cases
                ag_deg = np.zeros((len(net_seeds)*len(np_seeds),num_days))
                m_combo_alpha_results = m_combo_results[m_combo_results['alpha']==alpha]
                ai+=1
                #go through all net seeds
                ni=-1 #net index
                for net_seed in net_seeds:
                    m_combo_alpha_net_results = m_combo_alpha_results[m_combo_alpha_results['net_seed']==net_seed]
                    #go through all np seeds
                    for np_seed in np_seeds:
                        ni+=1
                        m_combo_alpha_net_np_results = m_combo_alpha_net_results[m_combo_alpha_net_results['np_seed']==np_seed]
                        ag_deg[ni,:] = np.array(m_combo_alpha_net_np_results['perc_above_left']) #Get deaths for combo and ag


                #Cumulative
                ag_deg_av = 100*np.average(ag_deg,axis=0)
                ag_deg_std =100*(sem(ag_deg,axis=0))


                #Add to total
                total[ci,ai,:,:] +=ag_deg_av

        #Total
        ci=0
        for c in combos:
            fig1, ax1 = plt.subplots(figsize=(4.5/2.54, 4/2.54))
            print(m)
            ai=0
            for a in alphas:
                m_deg_av = np.average(total[ci,ai,:,:],axis=0)
                m_deg_std = sem(total[ci,ai,:,:],axis=0)
                ax1.plot(np.arange(m_deg_av.shape[0]), m_deg_av, color = colors[str(a)], linewidth=1, label = str(a))
                ax1.fill_between(np.arange(m_deg_av.shape[0]),m_deg_av-m_deg_std,m_deg_av+m_deg_std,color = colors[str(a)],alpha=0.5)

                ai+=1
            ci+=1

            ax1.set_xticks(x_dates)
            ax1.set_xticklabels(dates, rotation='vertical')
            ax1.set_title('m='+str(m)+'|'+labels[c])
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.set_ylabel('% above t left')
            fig1.tight_layout()
            fig1.savefig(outdir+'deg_'+str(m)+'_'+str(c)+'_total.png', format='png', dpi=300)
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
epidemic_data = pd.read_csv(datadir+'Spain.csv')
epidemic_data= epidemic_data.loc[:223] #11 feb to 21 Sep mobility data exists
observed_deaths = np.array(np.flip(epidemic_data['deaths']))
sm_deaths = np.zeros(observed_deaths.shape[0])
#Smooth the deaths
#First go through the deaths and set the negative ones to the varege of that before and after
for i in range(len(observed_deaths)):
    if observed_deaths[i]<0:
        observed_deaths[i] = (observed_deaths[i-1]+observed_deaths[i+1])/2
#Do a 7day sliding window to get more even death predictions
for i in range(7,len(sm_deaths)+1):
    sm_deaths[i-1]=np.average(observed_deaths[i-7:i])
sm_deaths[0:6] = sm_deaths[6] #assign the first week

#Age groups
age_groups = ['0-19','20-49','50-69','70+']
#Get the results
result_dfs = glob.glob(resultsdir+'*.csv')
#Loop through all results dfs
all_results = pd.DataFrame()
combos = {'1_1_1_1':1, '2_2_2_2':2, '4_4_4_4':3, '3_3_3_3':4}

for name in result_dfs:
    resultdf = pd.read_csv(name)
    num_days = len(resultdf)
    print(num_days)
    info = name.split('/')[-1][:-4]
    m = int(info.split('_')[1])
    resultdf['m']=m
    resultdf['net_seed']=int(info.split('_')[2])
    resultdf['np_seed']=int(info.split('_')[3])
    resultdf['combo']='_'.join(info.split('_')[-5:-1])
    resultdf['alpha']=info.split('_')[-1]

    #append df
    all_results = all_results.append(resultdf)


#xticks
x_dates = [  0,  28,  56,  84, 112, 140, 168, 196, 224]
dates = ['Feb 9', 'Mar 8', 'Apr 5','May 3', 'May 31', 'Jun 28','Jul 26','Aug 23', 'Sep 21']
colors = {'1.0':'grey', '2.0':'g','3.0':'cornflowerblue'}
labels = {'1_1_1_1':'0-49: 100%,50+: 100%', '2_2_2_2':'0-49: 50%,50+: 50%', '3_3_3_3':'0-49: 33%,50+: 33%', '4_4_4_4':'0-49: 25%,50+: 25%'}

#Plot deaths
#Plot deaths
#plot_deaths(all_results, age_groups, num_days, sm_deaths, n, x_dates, dates,colors, labels, outdir+'deaths/')

#Plot cases
#plot_cases(all_results, age_groups, num_days, n, colors, labels, x_dates, dates, outdir+'cases/')

#Plot the max degree reomved each day
plot_degrees(all_results, age_groups, num_days, n, colors, labels, x_dates, dates, outdir+'degrees/')
#Plot the number removed - the ones that have issued spread
#plot_epidemic(np.arange(num_days), 100*np.array(num_removed)/n,'Days since initial spread','% Active spreaders','Active spreaders',m, outdir+'active_spreaders_'+str(m)+suffix)
