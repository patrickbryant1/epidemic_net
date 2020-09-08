#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gamma, lognorm
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

def conv_lognorm_params(mean,std):
        '''Returns converted shape and scale params
        A common parametrization for a lognormal random variable Y is in terms of the mean, mu, and standard deviation, sigma,
        of the unique normally distributed random variable X such that exp(X) = Y.
        This parametrization corresponds to setting s = sigma and scale = exp(mu).
        lognorm.pdf(x, s, loc, scale)
        '''
        shape = std
        scale = np.exp(mean)

        return shape,scale

def infection_to_death():
        '''Simulate the time from infection to death: Infection --> Onset --> Death'''
        #Infection to death: sum of ito and otd
        itd_shape, itd_scale = conv_gamma_params((5.1+17.8), (0.45))
        itd = gamma(a=itd_shape, scale = itd_scale) #a=shape
        return itd

def serial_interval_distribution(N):
        '''Models the the time between when a person gets infected and when
        they subsequently infect another other people
        '''
        serial_shape, serial_scale = conv_lognorm_params(4.7,2.9) #https://www.sciencedirect.com/science/article/pii/S1201971220301193
        serial = lognorm(s=serial_shape, scale = serial_scale) #a=shape

        return serial.pdf(np.arange(1,N+1))

###Data formatting
def read_and_format_data(datadir, outdir):
        '''Read in and format all data needed for the model
        N = number of days to model
        '''
        N=300

        #Get epidemic data
        epidemic_data = pd.read_csv(datadir+'stockholm.csv')

        #Mobility data
        #mobility_data = pd.read_csv(datadir+'Global_Mobility_Report.csv')
        #Convert to datetime
        #mobility_data['date']=pd.to_datetime(mobility_data['date'], format='%Y/%m/%d')

        #SI
        serial_interval = serial_interval_distribution(N) #pd.read_csv(datadir+"serial_interval.csv")
        #Infection fatality rate
        ifr=0.0058
        #Infection to death distribution
        itd = infection_to_death()
        #Get hazard rates for all days in country data
        h = np.zeros(N)
        f = np.cumsum(itd.pdf(np.arange(1,len(h)+1,0.5))) #Cumulative probability to die for each day
        #Adjust f to reach max 1 - the half steps makes this different
        f = f/2
        for i in range(1,len(h)):
            #for each day t, the death prob is the area btw [t-0.5, t+0.5]
            #divided by the survival fraction (1-the previous death fraction), (fatality ratio*death prob at t-0.5)
            #This will be the percent increase compared to the previous end interval
            h[i] = (ifr*(f[i*2+1]-f[i*2-1]))/(1-ifr*f[i*2-1])

        #The number of deaths today is the sum of the past infections weighted by their probability of death,
        #where the probability of death depends on the number of days since infection.
        s = np.zeros(N)
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
        plt.savefig(outdir+'f.png')
        plt.close()
        plt.plot(np.arange(len(serial_interval)),serial_interval)
        plt.savefig(outdir+'SI.png')

        return serial_interval, f

def simulate(serial_interval, f):
        '''Simulate epidemic development on a graph network.
        '''
        #Network
        n = 10000 #2385643, number of nodes
        m = 5 #Number of edges to attach from a new node to existing nodes
        Graph = nx.barabasi_albert_graph(n,m)
        edges = np.array(Graph.edges) #shape=n,2



        #Initial nodes
        num_initial = 10
        picked_nodes = np.random.choice(n, num_initial)
        #Number of days
        num_days=100

        #Susceptible
        #The susceptible population is the remaining edges (nodes in the edges)
        #Infected
        I = []
        inf_days = np.zeros(1000, dtype='int32') #Keep track of the infection days for each group
        #Removed
        R = []

        #Initial infection
        I.append(picked_nodes)
        #Simulate by connecting to the initial pick
        num_days=100
        for d in range(num_days):
            #Loop through the infection groups
            for i in range(len(I)):
                igroup = I[i] #Get the infected group
                inf_days[i]+=1 #Add one day to the infection group
                inf_prob = len(igroup)*np.sum(serial_interval[:inf_days[i]]) #Probability of the selected nodes to be infected
                inf_nodes = int(inf_prob) #Need to reach >0.5 to spread the infection
                if inf_nodes>0: #If there are nodes that can spread the infection
                    spread = np.random.choice(len(igroup),inf_nodes)
                    spread_nodes = igroup[spread]
                    R.append(spread_nodes) #Remove the nodes that have issued their spread
                    #Nodes left in igroup
                    I[i] = np.setdiff1d(igroup, spread_nodes)
                    new_infections = np.array([])
                    for inode in spread_nodes: #Get spread connections
                        inode_connections = np.append(edges[np.where(edges[:,0]==inode)][:,1], edges[np.where(edges[:,1]==inode)][:,0])
                        new_infections = np.append(new_infections, inode_connections)
                        #Remove from edges
                        keep_i = np.where((edges[:,0]!=inode)&(edges[:,1]!=inode))
                        edges = edges[keep_i]
                    I.append(new_infections) #append

                else:
                    continue
                print(d, edges.shape[0])

                if len(I)>len(inf_days):
                    pdb.set_trace()

            #Calculate the deaths
            
        pdb.set_trace()

        return

#####MAIN#####
args = parser.parse_args()
datadir = args.datadir[0]
outdir = args.outdir[0]

#Read and format data
serial_interval, f = read_and_format_data(datadir, outdir)
#Simulate
out = simulate(serial_interval, f)
