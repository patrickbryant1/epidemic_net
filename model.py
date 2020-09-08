#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gamma
import numpy as np
import seaborn as sns
import networkx as nx
import pdb



#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Simulate the epidemic development of Stockholm on a graph network''')

parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to outdir.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to outdir.')

###FUNCTIONS###

###DISTRIBUTIONS###
def conv_gamma_params(mean,std):
        '''Returns converted shape and scale params
        shape (α) = 1/std^2
        scale (β) = mean/shape
        '''
        shape = 1/(std*std)
        scale = mean/shape

        return shape,scale

def infection_to_death():
        '''Simulate the time from infection to death: Infection --> Onset --> Death'''
        #Infection to death: sum of ito and otd
        itd_shape, itd_scale = conv_gamma_params((5.1+17.8), (0.45))
        itd = gamma(a=itd_shape, scale = itd_scale) #a=shape
        return itd

def serial_interval_distribution(N2):
        '''Models the the time between when a person gets infected and when
        they subsequently infect another other people
        '''
        serial_shape, serial_scale = conv_gamma_params(4.7,2.9) #https://www.sciencedirect.com/science/article/pii/S1201971220301193
        serial = gamma(a=serial_shape, scale = serial_scale) #a=shape

        return serial.pdf(np.arange(1,N2+1))

###Data formatting
def read_and_format_data(datadir):
        '''Read in and format all data needed for the model
        N2 = number of days to model
        '''

        #Get epidemic data
        epidemic_data = pd.read_csv(datadir+'stockholm.csv')

        #Mobility data
        #mobility_data = pd.read_csv(datadir+'Global_Mobility_Report.csv')
        #Convert to datetime
        #mobility_data['date']=pd.to_datetime(mobility_data['date'], format='%Y/%m/%d')

        #SI
        serial_interval = serial_interval_distribution(N2) #pd.read_csv(datadir+"serial_interval.csv")

        #Infection to death distribution
        itd = infection_to_death()
        #Get hazard rates for all days in country data
        h = np.zeros(N2) #N2 = N+forecast
        f = np.cumsum(itd.pdf(np.arange(1,len(h)+1,0.5))) #Cumulative probability to die for each day
        #Adjust f to reach max 1 - the half steps makes this different
        f = f/2
        for i in range(1,len(h)):
            #for each day t, the death prob is the area btw [t-0.5, t+0.5]
            #divided by the survival fraction (1-the previous death fraction), (fatality ratio*death prob at t-0.5)
            #This will be the percent increase compared to the previous end interval
            h[i] = (cfr*(f[i*2+1]-f[i*2-1]))/(1-cfr*f[i*2-1])

        #The number of deaths today is the sum of the past infections weighted by their probability of death,
        #where the probability of death depends on the number of days since infection.
        s = np.zeros(N2)
        s[0] = 1
        for i in range(1,len(s)):
            #h is the percent increase in death
            #s is thus the relative survival fraction
            #The cumulative survival fraction will be the previous
            #times the survival probability
            #These will be used to track how large a fraction is left after each day
            #In the end all of this will amount to the adjusted death fraction
            s[i] = s[i-1]*(1-h[i-1]) #Survival fraction

        #Multiplying s and h yields fraction dead of fraction survived
        f = s*h #This will be fed to the Stan Model
        plt.plot(np.arange(len(f)),f)
        plt.show()
        pdb.set_trace()

        #Network


def simulate():
        '''Simulate epidemic development on a graph network.
        '''


        return out

#####MAIN#####
args = parser.parse_args()
datadir = args.datadir[0]
outdir = args.outdir[0]

#Read and format data
read_and_format_data(datadir)
#Simulate
out = simulate(stan_data, stan_model, outdir)
