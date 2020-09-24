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
import matplotlib.pyplot as plt
import matplotlib
import pdb



#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Simulate the epidemic development of New York on a graph network''')

parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to datadir.')
parser.add_argument('--n', nargs=1, type= int, default=sys.stdin, help = 'Num nodes in net.')
parser.add_argument('--m', nargs=1, type= int, default=sys.stdin, help = 'Num links to add for each new node in the preferential attachment graph.')
parser.add_argument('--graph_type', nargs=1, type= str, default=sys.stdin, help = 'Graph type: random or preferential_attachment.')
parser.add_argument('--s', nargs=1, type= str, default=sys.stdin, help = 'Spread reduction. Float to multiply infection probability with.')
parser.add_argument('--alpha', nargs=1, type= float, default=sys.stdin, help = 'Float to multiply mobility impact with.')
parser.add_argument('--num_initial', nargs=1, type= int, default=sys.stdin, help = 'Num initial nodes in net.')
parser.add_argument('--pseudo_count', nargs=1, type= int, default=sys.stdin, help = 'Pseudo count (number of nodes).')
parser.add_argument('--net_seed', nargs=1, type= int, default=sys.stdin, help = 'Seed for random initializer of network graph.')
parser.add_argument('--np_seed', nargs=1, type= int, default=sys.stdin, help = 'Seed for random initializer of network graph.')
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

        #Get epidemic data
        epidemic_data = pd.read_csv(datadir+'new_york.csv')

        #Convert to datetime
        epidemic_data['date']=pd.to_datetime(epidemic_data['DATE_OF_INTEREST'])

        #Mobility data
        mobility_data = pd.read_csv(datadir+'Global_Mobility_New_York.csv')
        #Convert to datetime
        mobility_data['date']=pd.to_datetime(mobility_data['date'], format='%Y/%m/%d')
        #Join epidemic data and mobility data
        epidemic_data = pd.merge(epidemic_data,mobility_data, left_on = 'date', right_on ='date', how = 'right')
        epidemic_data = epidemic_data.dropna()
        N=len(epidemic_data)+11 #11 extra days

        #SI
        serial_interval = serial_interval_distribution(N)
        #Infection fatality rate
        ifr_by_age_group = {'0-19':0.00003,'20-49':0.0002,'50-69':0.005,'70+':0.054}

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


        #Mobility data
        #Covariates (mobility data from Google)
        #Construct a 1-week sliding average to smooth the mobility data

        mob_sectors = ['retail_and_recreation_percent_change_from_baseline',
       'grocery_and_pharmacy_percent_change_from_baseline',
       'parks_percent_change_from_baseline',
       'transit_stations_percent_change_from_baseline',
       'workplaces_percent_change_from_baseline',
       'residential_percent_change_from_baseline']
        mob_labels = {'retail_and_recreation_percent_change_from_baseline':'Retail',
                      'grocery_and_pharmacy_percent_change_from_baseline':'Grocery',
                      'parks_percent_change_from_baseline':'Parks',
                      'transit_stations_percent_change_from_baseline':'Transit',
                      'workplaces_percent_change_from_baseline':'Work',
                      'residential_percent_change_from_baseline':'Residential'
                      }
        y = np.zeros((len(mob_sectors),len(epidemic_data)))
        fig, ax = plt.subplots(figsize=(4.5/2.54, 4/2.54))
        x_dates = [  0,  28,  56,  84, 112, 140, 168, 197]
        dates = ['Feb 29', 'Mar 28', 'Apr 25','May 23', 'Jun 20','Jul 18','Aug 15', 'Sep 13']

        for m in range(len(mob_sectors)):
            sector = mob_sectors[m]
            mob_i = np.array(epidemic_data[sector])

            for i in range(7,len(mob_i)+1):
                #Check that there are no NaNs
                if np.isnan(mob_i[i-7:i]).any():
                    #If there are NaNs, loop through and replace with value from prev date
                    for i_nan in range(i-7,i):
                        if np.isnan(mob_i[i_nan]):
                            mob_i[i_nan]=mob_i[i_nan-1]
                y[m,i-1]=np.average(mob_i[i-7:i])#Assign average
            y[m,0:6] = y[m,6]#Assign first week
            plt.plot(np.arange(y.shape[1]-11),y[m,:][11:], label = mob_labels[sector], linewidth=1)
        #Reverse residential
        y[5,:] = -y[5,:]
        plt.plot(np.arange(y.shape[1]-11),np.average(y,axis=0)[11:], label = 'Average', linewidth=2, color = 'k')
        plt.title('New York mobility')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(x_dates)
        ax.set_xticklabels(dates, rotation='vertical')
        plt.ylabel('Mobility change')
        plt.tight_layout()
        fig.savefig('mobility.png', format='png', dpi=300)
        plt.close()

        mob_data = np.zeros(N)
        mob_data[11:]=np.average(y,axis=0)
        pdb.set_trace()
        return serial_interval, f, N, mob_data

def simulate(graph_type,serial_interval, f, N, outdir, n, m, mob_data, spread_reduction,alpha,num_initial,pseudo_count,net_seed, np_seed):
        '''Simulate epidemic development on a graph network.
        '''
        #Network
        if graph_type == 'random':
            Graph = nx.gnp_random_graph(n,(1/(5000/m)),seed=net_seed)
        if graph_type == 'preferential_attachment':
            Graph = nx.barabasi_albert_graph(n,m,seed=net_seed)

        degrees = np.array(Graph.degree)[:,1]
        edges = np.array(Graph.edges) #shape=n,2
        #Save edges
        # outname = outdir+str(n)+'_'+str(m)+'_'+str(seed)
        #
        # for s in [*spread_reduction.values()]:
        #     outname+='_'+str(s)
        #np.save(outname+'_edges.npy', edges)

        #Population
        age_groups = ['0-19','20-49','50-69','70+']
        population_shares = [0.23,0.45,0.22,0.10]
        #Lockdown 22 March: https://www.bbc.com/news/world-us-canada-52757150
        #Don't have to worry about this - will be taken care of by mobility relation
        #Epidemic starts 28 days before 10 cumulative deaths = 28 days before 17 March = 18 Feb
        #There are thus 33 days (28+5) until Lockdown
        #The data starts at 29th feb --> have to remove 11 days in the beginning


        #Assign the nodes randomly according to the population shares
        ag_nodes = {}#Nodes per age group
        not_chosen = np.arange(n)
        ps = 0
        for ag in age_groups:
            chosen = np.random.choice(not_chosen, int(n*population_shares[ps]),replace=False)
            ag_nodes[ag] = chosen
            not_chosen = np.setdiff1d(not_chosen,chosen)
            ps+=1#Increas population share index

        #Initial nodes
        #num_initial = 1 #represents start at num_initial*2385643/n
        initial_infections = np.random.choice(n, num_initial,replace=False)
        #Number of days
        num_days=N
        #Pseudo count
        #pseudo_count = 1
        #Susceptible
        S = np.arange(n)
        #Infected
        I = []
        #Removed
        R = []
        num_removed = [0]
        dg_of_removed = [[0]] #Keep track of the degrees of the removed nodes
        #Keep track of remaining edges
        remaining_edges = [edges.shape[0]]
        #Initial infection
        I.extend(initial_infections)
        num_infected_day = [num_initial]
        num_new_infections = [num_initial]
        num_new_infections_age_group = {'0-19':[],'20-49':[],'50-69':[],'70+':[]}

        #Add the initial infections per age group
        for ag in num_new_infections_age_group:
            num_new_infections_age_group[ag].append(len(initial_infections[np.isin(initial_infections,ag_nodes[ag])]))
        #Remove the initially picked nodes from S
        S = np.setdiff1d(S,initial_infections)
        #Simulate by connecting to the initial pick
        #The other model saves all infections and use them for multiplication with the SI
        print('day edges num_spread_nodes num_infected num_new_infections num_removed num_new_removed')
        for d in range(1,num_days):
            prevR = len(R) #The number of removed the previous day
            new_infections=[]
            new_degrees = [] #Save the degrees of the removed nodes for day d
            #Loop through all days up to current to get infection probability
            #This probability should be based on the total number of infections on a given day.
            inf_prob = 0 #Infection probability at day d
            for prev_day in range(d):
                inf_prob += num_infected_day[prev_day]*serial_interval[d-prev_day] #Probability of the selected nodes to spread the infection

            #Spread infection
            #Save the new infections
            new_infections = np.array([])

            inf_nodes = int(np.round(inf_prob)) #Need to reach >0.5 to spread the infection
            if inf_nodes>0 and len(I)>0: #If there are nodes that can spread the infection
                if len(I)>inf_nodes:
                    spread_indices = np.random.choice(len(I), inf_nodes,replace=False)
                else:
                    spread_indices = np.arange(len(I))
                spread_nodes = np.array(I)[spread_indices]

                #Get the new infections
                selected_spread_nodes = []
                for inode in spread_nodes: #Get spread connections

                    inode_connections = np.append(edges[np.where(edges[:,0]==inode)][:,1], edges[np.where(edges[:,1]==inode)][:,0])
                    selected_spread_nodes.append(inode) #Save to spread nodes
                    #Check that there are new connections (not isolated node - surrounding infected)
                    if len(inode_connections)>0:
                        selected_connections = []
                        for connection in inode_connections:
                            for ag in ag_nodes: #Check age group
                                if connection in ag_nodes[ag]:
                                    if spread_reduction[ag]!=1: #If the reduction is 1, all spread should happen
                                        #See if node should be infected based on reduction in infection probability
                                        if np.random.randint(spread_reduction[ag])==spread_reduction[ag]-1:
                                            selected_connections.append(connection)

                                    else:
                                        selected_connections.append(connection)
                        #Set new connections based on reduced infection probabilities
                        inode_connections = np.array(selected_connections)

                        new_infections = np.append(new_infections, inode_connections)
                        #Remove infectious node from edges
                        edges = edges[edges[:,0]!=inode]
                        edges = edges[edges[:,1]!=inode]
                        R.append(inode)



                    #Save the dg of the removed node - even if it is isolated
                    #The degree is not necessarily the same as the number of connections
                    #If links to the node are removed - its degree is reduced
                    new_degrees.append(len(inode_connections))


                #Remove the spread nodes from I (no longer infectious after they issued their spread)
                I = [*np.setdiff1d(I, selected_spread_nodes)]
                #Get only the unique nodes in the new infections
                new_infections = np.unique(new_infections)


                #Check if the new infections are in the S - otherwise the nodes may already be infected
                new_infections = new_infections[np.isin(new_infections,S)]

                #Remove from S
                S = np.setdiff1d(S,new_infections)



            #Save the degrees for day d
            dg_of_removed.append(new_degrees)

            if len(S)<1:
                num_new_infections.append(len(new_infections))
                num_infected_day.append(len(I))
                num_removed.append(len(R)-prevR) #The difference will be the nodes that have issued their infection
                remaining_edges.append(edges.shape[0])
                #Add the new infections per age group (if there are any)
                for ag in num_new_infections_age_group:
                    if len(new_infections)>0:
                        num_new_infections_age_group[ag].append(len(new_infections[np.isin(new_infections,ag_nodes[ag])]))
                    else:
                        num_new_infections_age_group[ag].append(0)
                print(d, remaining_edges[d], inf_nodes, num_infected_day[d],num_new_infections[d],len(R), num_removed[d])
                continue
            #Add infection from new node
            try:
                new_node = np.random.choice(S,pseudo_count)
            except:
                pdb.set_trace()
            new_infections = np.append(new_infections,new_node)
            S = np.setdiff1d(S,new_node)
            I.extend(new_infections) #add to new infections
            num_infected_day.append(len(I))
            num_new_infections.append(len(new_infections))
            #Add the new infections per age group (if there are any)
            for ag in num_new_infections_age_group:
                if len(new_infections)>0:
                    num_new_infections_age_group[ag].append(len(new_infections[np.isin(new_infections,ag_nodes[ag])]))
                else:
                    num_new_infections_age_group[ag].append(0)

            num_removed.append(len(R)-prevR) #The difference will be the nodes that have issued their infection
            remaining_edges.append(edges.shape[0])
            print(d, remaining_edges[d], inf_nodes, num_infected_day[d],num_new_infections[d],len(R), num_removed[d])
            #Dynamic features - reconnect edges
            #Reduce the inf prob according to the mob data
            m_edges = 0.1*(len(edges))*np.exp(alpha*mob_data[d-1]/100)
            if len(edges)>0:
                edges = reconnect(edges,m_edges)
        #Calculate deaths
        deaths = np.zeros((f.shape[0],num_days))
        for ai in range(deaths.shape[0]):
            ag = age_groups[ai]
            for di in range(1,num_days): #Loop through all days
                for dj in range(di): #Integrate by summing the num_removed*f[]
                    deaths[ai,di] += num_new_infections_age_group[ag][dj]*f[ai,di-dj]


        #Format the degrees of the removed nodes (those that have issued spread)
        x_times_av_dg = np.average(degrees)*5
        num_above_left = [] #Save the number of nodes with a degree of at least x_times_av_dg left in the net
        num_above = degrees[degrees>=x_times_av_dg].shape[0]
        for dgs in dg_of_removed:
            if len(dgs)>0:
                for dg in dgs:
                    if dg >= x_times_av_dg:
                        num_above-=1

            num_above_left.append(num_above)

        #Save results
        result_df = pd.DataFrame()
        result_df['day'] = np.arange(num_days)
        result_df['edges'] = remaining_edges
        result_df['num_infected'] = num_infected_day
        result_df['num_new_infections'] = num_new_infections
        result_df['num_new_removed'] = num_removed
        for ai in range(deaths.shape[0]):
            result_df[age_groups[ai]+' deaths'] = deaths[ai,:]
            result_df[age_groups[ai]+' cases']=num_new_infections_age_group[age_groups[ai]]
        result_df['perc_above_left']=np.array(num_above_left)/max(num_above_left)
        outname = outdir+'results_'+str(m)+'_'+str(net_seed)+'_'+str(np_seed)
        for s in [*spread_reduction.values()]:
            outname+='_'+str(s)
        outname += '_'+str(alpha)
        result_df.to_csv(outname+'.csv')

        return None


def reconnect(edges,m):
    '''Simulate new connections by randomly connecting remaining edges.
    This will simulate movement - as of a dynamic network
    '''

    remaining_nodes = np.unique(edges)
    def get_new_edge(remaining_nodes):
        '''Get a new edge
        '''
        #Choose 2 random nodes to reconnect
        new_edge = np.random.choice(remaining_nodes,2, replace=False)

        #Sort to get order as in edges
        new_edge = np.sort(new_edge)

        return new_edge


    #Get new edges
    num_new_edges = int(m)
    print(num_new_edges)
    new_edges = []
    fetched_edges = 0 #Edge index

    while fetched_edges < num_new_edges:
        new_edge = get_new_edge(remaining_nodes)

        #If the edge exists, continue (count it as being both removed and reconnected) - otherwise save it
        if len(np.where((edges[:,0]==new_edge[0]) & (edges[:,1]==new_edge[1]))[0])<1:
            fetched_edges+=1
            continue
        else:
            #Remove another edge
            remove_index = int(np.random.choice(remaining_nodes,1, replace=False)[0])
            edges = np.concatenate([edges[:remove_index],edges[remove_index+1:]])
            new_edges.append(new_edge)
            fetched_edges+=1




    #concat
    if len(new_edges)>1:
        edges =np.concatenate([edges,np.array(new_edges)])
    return edges



#####MAIN#####
matplotlib.rcParams.update({'font.size': 5.5})
args = parser.parse_args()
n = args.n[0]
m = args.m[0]
graph_type = args.graph_type[0]
s = args.s[0].split('_')
alpha = args.alpha[0]
num_initial = args.num_initial[0]
pseudo_count = args.pseudo_count[0]
datadir = args.datadir[0]
net_seed = args.net_seed[0]
np_seed = args.np_seed[0]
outdir = args.outdir[0]

#Seed np random seed
np.random.seed(np_seed)
#Initial reduction
spread_reduction =  {'0-19':1,'20-49':1,'50-69':1,'70+':1}
ai=0
for ag in spread_reduction:
    spread_reduction[ag] = int(s[ai])
    ai+=1

#Read and format data
serial_interval, f, N, mob_data = read_and_format_data(datadir, outdir)
#Simulate
print('Simulating',m)
simulate(graph_type,serial_interval, f, N, outdir, n, m, mob_data, spread_reduction,alpha,num_initial,pseudo_count,net_seed, np_seed)
