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
import networkx as nx
import pdb



#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Visualize graph networks''')

parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to outdir.')

###FUNCTIONS###
def visualize(outdir):
    '''Vizualize a network
    '''
    #Create a graph
    #gr = nx.Graph()
    #gr.add_edges_from(network)
    for m in range(1,4):
        gr = nx.barabasi_albert_graph(100,m,seed=0)
        edges = np.array(gr.edges)
        print(len(edges))
        fig, ax = plt.subplots(figsize=(4.5/2.54, 4/2.54))
        nx.draw(gr, width=0.5,node_size=1,node_color='lightsteelblue')
        ax.set_title('m='+str(m))
        fig.tight_layout()
        fig.savefig(outdir+str(m)+'_pa.png', format='png', dpi=300)
        plt.close()

    for m in range(1,4):
        gr = nx.gnp_random_graph(100,1/(50/m),seed=0)
        edges = np.array(gr.edges)
        print(len(edges))
        fig, ax = plt.subplots(figsize=(4.5/2.54, 4/2.54))
        nx.draw(gr, width=0.5,node_size=1,node_color='lightsteelblue')
        ax.set_title('p='+str(1/(50/m)))
        fig.tight_layout()
        fig.savefig(outdir+str(m)+'_random.png', format='png', dpi=300)
        plt.close()

    return None
#####MAIN#####
#Set font size
matplotlib.rcParams.update({'font.size': 5.5})
args = parser.parse_args()

outdir = args.outdir[0]

#Visualize
visualize(outdir)
# visualize(network1, outdir+'net1.png')
# visualize(network2, outdir+'net2.png')
# visualize(network3, outdir+'net3.png')
# visualize(network4, outdir+'net4.png')
# visualize(network5, outdir+'net5.png')
# pdb.set_trace()
