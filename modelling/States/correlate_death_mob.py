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
from scipy.stats import pearsonr

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

    #Convert to datetime
    mobility_data['date']=pd.to_datetime(mobility_data['date'], format='%Y/%m/%d')
    mob_sectors = ['retail_and_recreation_percent_change_from_baseline',
   'grocery_and_pharmacy_percent_change_from_baseline',
   'parks_percent_change_from_baseline',
   'transit_stations_percent_change_from_baseline',
   'workplaces_percent_change_from_baseline',
   'residential_percent_change_from_baseline']

    formatted_data = pd.DataFrame()
    #Get data by state
    subregions =  mobility_data['sub_region_1'].unique()[1:]
    for i in range(len(subregions)):

        #State
        region =subregions[i]
        #Get region epidemic data
        regional_deaths = us_deaths[us_deaths['Province_State']== region]
        cols = regional_deaths.columns
        #Calculate back per day - now cumulative
        deaths_per_day = []
        dates = cols[12:]
        #First deaths
        deaths_per_day.append(np.sum(regional_deaths[dates[0]])) #The first 12 columns are not deaths
        for d in range(1,len(dates)):
            deaths_per_day.append(np.sum(regional_deaths[dates[d]])-np.sum(regional_deaths[dates[d-1]]))

        #Create dataframe
        regional_epidemic_data = pd.DataFrame()
        regional_epidemic_data['date']=dates

        #Convert to datetime
        regional_epidemic_data['date'] = pd.to_datetime(regional_epidemic_data['date'], format='%m/%d/%y')
        regional_epidemic_data['deaths']=deaths_per_day

        #Sort on date
        regional_epidemic_data = regional_epidemic_data.sort_values(by='date')
        #Regional mobility data
        region_mob_data = mobility_data[mobility_data['sub_region_1']==region]
        region_mob_data = region_mob_data[region_mob_data['sub_region_2'].isna()]
        #Merge epidemic data with mobility data
        regional_epidemic_data = regional_epidemic_data.merge(region_mob_data, left_on = 'date', right_on ='date', how = 'right')

        #Smooth deaths
        #Number of deaths
        deaths = np.array(regional_epidemic_data['deaths'])
        sm_deaths = np.zeros(len(deaths))
        #Do a 7day sliding window to get more even death predictions
        for i in range(7,len(regional_epidemic_data)+1):
            sm_deaths[i-1]=np.average(deaths[i-7:i])
        sm_deaths[0:6] = sm_deaths[6]

        #Add to df
        regional_epidemic_data['deaths']=sm_deaths
        #Covariates (mobility data from Google) - assign the same shape as others (N2)
        #Construct a 1-week sliding average to smooth the mobility data
        regional_mob_data = []
        for name in mob_sectors:
            mob_i = np.array(regional_epidemic_data[name])
            y = np.zeros(len(regional_epidemic_data))
            for i in range(7,len(mob_i)+1):
                #Check that there are no NaNs
                if np.isnan(mob_i[i-7:i]).any():
                    #If there are NaNs, loop through and replace with value from prev date
                    for i_nan in range(i-7,i):
                        if np.isnan(mob_i[i_nan]):
                            mob_i[i_nan]=mob_i[i_nan-1]
                y[i-1]=np.average(mob_i[i-7:i])#Assign average
            y[0:6] = y[6]#Assign first week
            regional_epidemic_data[name]=y
            regional_mob_data.append(y)
        #Get deaths per population size
        regional_epidemic_data['deaths_per_population'] = sm_deaths/population_sizes[population_sizes['State']==region]['Population'].values[0]
        #Save to formatted data
        formatted_data = pd.concat([formatted_data, regional_epidemic_data])

    return formatted_data




def correlate(formatted_data):
    '''Correlate the mobility and deaths
    '''


    mob_sectors = {'retail_and_recreation_percent_change_from_baseline':'grey',
   'grocery_and_pharmacy_percent_change_from_baseline':'b',
   'parks_percent_change_from_baseline':'g',
   'transit_stations_percent_change_from_baseline':'orange',
   'workplaces_percent_change_from_baseline':'magenta',
   'residential_percent_change_from_baseline':'k'}

    subregions = formatted_data['sub_region_1'].unique()

    fig1, ax1 = plt.subplots(figsize=(16/2.54, 9/2.54))
    fig2, ax2 = plt.subplots(figsize=(16/2.54, 9/2.54))
    ax2_2 = ax2.twinx()
    fig3, ax3 = plt.subplots(figsize=(16/2.54, 9/2.54))
    for region in subregions:
        regional_data = formatted_data[formatted_data['sub_region_1']==region]
        deaths = np.array(regional_data['deaths_per_population'])
        #ax1.plot(np.arange(len(regional_data['date'])),np.log10(regional_data['deaths_per_population']), color = 'grey', alpha =0.1)
        regional_mob_data = []
        for sector in mob_sectors:
            regional_mob_data.append(regional_data[sector])
        regional_mob_data = np.array(regional_mob_data)
        #reverse resideantial
        regional_mob_data[5]=-regional_mob_data[5]
        #Average
        av_mob = np.average(regional_mob_data,axis=0)
        #ax2.plot(np.arange(len(regional_data['date'])),av_mob, color = 'k', alpha = 0.1)

        s_max = 100
        C_death_delay = np.zeros(s_max) #Save covariance btw signals for different delays in deaths
        #Loop through all s and calculate correlations
        for s in range(s_max): #s is the number of future days to correlate the death and mobility data over
            if s == 0:
                cs = pearsonr(av_mob,deaths)[0]

            else:
                cs = pearsonr(av_mob[:-s],deaths[s:])[0]
            #Assign correlation
            C_death_delay[s]=cs
        #Plot delay and PCC
        ax1.plot(np.arange(s_max),C_death_delay,linewidth=1)

        #Check the max corr
        state_max = np.where(C_death_delay==max(C_death_delay))[0][0]
        if state_max ==0:
            state_max=1
            print(region)

        ax2.plot(np.arange(len(av_mob)-state_max), av_mob[:-state_max], color= 'b',linewidth=1)
        ax2_2.plot(np.arange(len(av_mob)-state_max), np.log10(deaths[state_max:]), color = 'r', linewidth=1)

        #Plot mob vs deaths
        ax3.scatter(av_mob[:-state_max],np.log10(deaths[state_max:]), color= 'b',s=1)

        #Plot state
        fig, ax = plt.subplots(figsize=(4.5/2.54, 4/2.54))
        ax_x = ax.twinx()
        sm_deaths = np.array(regional_data['deaths'])
        zeros = np.zeros(len(sm_deaths)+state_max)
        zeros[:-state_max]=sm_deaths
        #Deaths
        ax.bar(np.arange(len(zeros)),zeros, color = 'grey')
        zeros = np.zeros(len(sm_deaths)+state_max)
        zeros[state_max:]=av_mob
        min_mob_i = np.where(zeros==min(av_mob))[0][0]-state_max
        ax_x.axvline(min_mob_i, color = 'r', linewidth=0.8)
        #Mobility
        ax_x.plot(np.arange(len(zeros)),zeros, color = 'k', linewidth = 1)
        ax.set_ylabel('Deaths')
        ax_x.set_ylabel('Mobility')
        ax_x.set_ylim([-40,40])
        ax.set_title(region+'|'+str(state_max))
        fig.tight_layout()
        ax.spines['top'].set_visible(False)
        ax_x.spines['top'].set_visible(False)
        fig.savefig(outdir+region+'.png', dpi=300, format='png')


    ax1.set_xlabel('Delay in deaths (days)')
    ax1.set_ylabel('PCC')
    fig1.tight_layout()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    fig1.savefig(outdir+'corr_delay', dpi=300, format='png')

    ax2.set_xlabel('Day')
    ax2.set_ylabel('Mobility')
    ax2_2.set_ylabel('log(Deaths/Population)')
    fig2.tight_layout()
    ax2.spines['top'].set_visible(False)
    ax2_2.spines['top'].set_visible(False)
    fig2.savefig(outdir+'mob_deaths', dpi=300, format='png')

    ax3.set_xlabel('Mobility')
    ax3.set_ylabel('Deaths/Population')
    fig3.tight_layout()
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    fig3.savefig(outdir+'mob_vs_deaths', dpi=300, format='png')

    montage = ''
    for region in subregions:
        montage += region+ '.png '
    print(montage)
#####MAIN#####
#Set font size
matplotlib.rcParams.update({'font.size': 7})
args = parser.parse_args()
us_deaths = pd.read_csv(args.us_deaths[0])
mobility_data = pd.read_csv(args.mobility_data[0])
population_sizes = pd.read_csv(args.population_sizes[0])
outdir = args.outdir[0]

#Format data
try:
    formatted_data = pd.read_csv(outdir+'formatted_data.csv')
except:
    formatted_data = format_data(us_deaths, mobility_data, population_sizes)
    formatted_data.to_csv(outdir+'formatted_data.csv')

#Correlate
correlate(formatted_data)
