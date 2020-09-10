#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import glob
import pandas as pd
from scipy.stats import gamma, lognorm
import numpy as np
import networkx as nx
import pdb



#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Simulate the epidemic development of Stockholm on a graph network''')

parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to datadir.')
parser.add_argument('--n', nargs=1, type= int, default=sys.stdin, help = 'Num nodes in net.')
parser.add_argument('--m', nargs=1, type= int, default=sys.stdin, help = 'Num links to add for each new node in the preferential attachment graph.')
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
        stockholm_csv = pd.read_csv(datadir+'stockholm.csv')
        N=len(stockholm_csv)*7

        #Get epidemic data
        epidemic_data = pd.read_csv(datadir+'stockholm.csv')

        #Mobility data
        #mobility_data = pd.read_csv(datadir+'Global_Mobility_Report.csv')
        #Convert to datetime
        #mobility_data['date']=pd.to_datetime(mobility_data['date'], format='%Y/%m/%d')

        #SI
        serial_interval = serial_interval_distribution(N) #pd.read_csv(datadir+"serial_interval.csv")
        #Infection fatality rate
        av_ifr=0.0058
        ifr_by_age_group = {'Average':0.0058,'0-49':0.0001,'50-59':0.0027,'60-69':0.0045,'70-79':0.0192,'80-89':0.072,'90+':0.1621}
        #Infection to death distribution
        itd = infection_to_death()
        #Get hazard rates for all days in country data
        h = np.zeros((len(ifr_by_age_group),N))
        f = np.cumsum(itd.pdf(np.arange(1,N+1,0.5))) #Cumulative probability to die for each day
        #Adjust f to reach max 1 - the half steps makes this different
        f = f/2
        #Hazard rate per age group
        hi = 0
        for age in ifr_by_age_group:
            ifr = ifr_by_age_group[age]
            for i in range(1,N):
                #for each day t, the death prob is the area btw [t-0.5, t+0.5]
                #divided by the survival fraction (1-the previous death fraction), (fatality ratio*death prob at t-0.5)
                #This will be the percent increase compared to the previous end interval
                h[hi,i] = (ifr*(f[i*2+1]-f[i*2-1]))/(1-ifr*f[i*2-1])
            hi+=1 #Increase hazard index

        #The number of deaths today is the sum of the past infections weighted by their probability of death,
        #where the probability of death depends on the number of days since infection.
        s = np.zeros((len(ifr_by_age_group),N))
        s[:,0] = 1
        for agei in range(s.shape[0]):
            for i in range(1,s.shape[1]):
                #h is the percent increase in death
                #s is thus the relative survival fraction
                #The cumulative survival fraction will be the previous
                #times the survival probability
                #These will be used to track how large a fraction is left after each day
                #In the end all of this will amount to the adjusted death fraction
                s[agei,i] = s[agei,i-1]*(1-h[agei,i-1]) #Survival fraction

        #Multiplying s and h yields fraction dead of fraction survived
        f = s*h #This will be fed to the Stan Model

        # age_keys =[*ifr_by_age_group.keys()]
        # colors = ['royalblue', 'navy','lightskyblue', 'darkcyan', 'mediumseagreen', 'paleturquoise' ]
        # import matplotlib
        # import matplotlib.pyplot as plt
        # matplotlib.rcParams.update({'font.size': 7})
        # fig, ax = plt.subplots(figsize=(6.5/2.54, 4.5/2.54))
        # for fi in range(len(f)):
        #     plt.plot(np.arange(N),f[fi,:],label=age_keys[fi], color = colors[fi])
        # plt.legend()
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.set_title('Fatality rate')
        # ax.set_xlabel('Infection day')
        # ax.set_ylabel('Probability')
        # fig.tight_layout()
        # plt.savefig(outdir+'fatality_rate.png', format='png', dpi=300)
        # plt.close()
        # pdb.set_trace()
        # plt.plot(np.arange(len(serial_interval)),serial_interval)
        # plt.savefig(outdir+'SI.png')
        # plt.close()
        return serial_interval, f, N

def simulate(serial_interval, f, N, outdir, n, m):
        '''Simulate epidemic development on a graph network.
        '''
        #Network
        #n = 2385643 # number of nodes
        #m = 5 #Number of edges to attach from a new node to existing nodes - should be varied
        Graph = nx.barabasi_albert_graph(n,m)
        edges = np.array(Graph.edges) #shape=n,2
        #Save edges
        np.save(outdir+str(n)+'_'+str(m)+'_edges.npy', edges)


        #Initial nodes
        num_initial = 10
        initial_infections = np.random.choice(n, num_initial)
        #Number of days
        num_days=N

        #Susceptible
        S = np.arange(n)
        #Infected
        I = []
        #Removed
        R = []
        num_removed = [0]
        death_days = np.zeros(1000, dtype='int32') #Keep track of the infection days for each group
        #Keep track of remaining edges
        remaining_edges = [edges.shape[0]]
        #Initial infection
        I.extend(initial_infections)
        num_infected_day = [num_initial]
        num_new_infections = [num_initial]
        #Remove the initially picked nodes from S
        S = np.setdiff1d(S,initial_infections)
        #Simulate by connecting to the initial pick
        #The other model saves all infections and use them for multiplication with the SI
        print('day edges num_spread_nodes num_infected num_new_infections num_removed num_new_removed')
        for d in range(1,num_days):
            prevR = len(R) #The number of removed the previous day
            new_infections=[]
            #Loop through all days up to current to get infection probability
            #This probability should be based on the total number of infections on a given day.
            #I should count all the infections on day d and then get the inf_nodes from total_inf*np.sum(serial_interval[:inf_days[i]])
            #inf_nodes can then be chosen from all nodes on day d randomly.
            #Is there problems with selecting this way? The nodes selected earlier should have lower probability of being selected
            #for spread later in the epidemic. This way the probability may increase (?). Although with more infectious nodes
            #the probability of selection likely goes down.
            inf_prob = 0 #Infection probability at day d
            for prev_day in range(d):
                inf_prob += num_infected_day[prev_day]*serial_interval[d-prev_day] #Probability of the selected nodes to spread the infection

            #Spread infection
            inf_nodes = int(np.round(inf_prob)) #Need to reach >0.5 to spread the infection
            if inf_nodes>0 and len(I)>0: #If there are nodes that can spread the infection
                spread_indices = np.random.choice(len(I), inf_nodes)
                spread_nodes = np.array(I)[spread_indices]
                #Remove the spread nodes from I (no longer infectious after they issued their spread)
                I = [*np.setdiff1d(I, spread_nodes)]
                #Get the new infections
                new_infections = np.array([])
                for inode in spread_nodes: #Get spread connections
                    inode_connections = np.append(edges[np.where(edges[:,0]==inode)][:,1], edges[np.where(edges[:,1]==inode)][:,0])
                    #Check that there are new connections (not isolated node - surrounding infected)
                    if len(inode_connections)>0:
                        new_infections = np.append(new_infections, inode_connections)
                        #Remove from edges
                        edges = edges[edges[:,0]!=inode]
                        edges = edges[edges[:,1]!=inode]
                        R.append(inode)

                #Get only the unique nodes in the new infections
                new_infections = np.unique(new_infections)

                #Check if the new infections are in the S - otherwise the nodes may already be infected
                new_infections = new_infections[np.isin(new_infections,S)]
                #Remove from S
                S = np.setdiff1d(S,new_infections)
                I.extend(new_infections) #append

            num_infected_day.append(len(I))
            num_new_infections.append(len(new_infections))
            num_removed.append(len(R)-prevR) #The difference will be the nodes that have issued their infection
            remaining_edges.append(edges.shape[0])
            print(d, remaining_edges[d], inf_nodes, num_infected_day[d],num_new_infections[d], len(R), num_removed[d])


        num_new_infections = np.array(num_new_infections)
        #Calculate deaths
        age_groups = ['Average','0-49','50-59','60-69','70-79','80-89','90+']
        population_shares = [1,0.666,0.125,0.092,0.077,0.032,0.008]
        deaths = np.zeros((f.shape[0],num_days))
        for ai in range(deaths.shape[0]):
            for di in range(1,num_days): #Loop through all days
                for dj in range(di): #Integrate by summing the num_removed*f[]
                    deaths[ai,di] += num_new_infections[dj]*population_shares[ai]*f[ai,di-dj]

        #Save results
        result_df = pd.DataFrame()
        result_df['day'] = np.arange(num_days)
        result_df['edges'] = remaining_edges
        result_df['num_infected'] = num_infected_day
        result_df['num_new_infections'] = num_new_infections
        result_df['num_new_removed'] = num_removed
        for ai in range(deaths.shape[0]):
            result_df[age_groups[ai]+' deaths'] = deaths[ai,:]
        result_df.to_csv(outdir+'results_'+str(m)+'.csv')

        return None



#####MAIN#####
args = parser.parse_args()
n = args.n[0]
m = args.m[0]
datadir = args.datadir[0]
outdir = args.outdir[0]

#Read and format data
serial_interval, f, N = read_and_format_data(datadir, outdir)
#Simulate
simulate(serial_interval, f, N, outdir, n, m)
