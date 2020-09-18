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

parser.add_argument('--network1', nargs=1, type= str, default=sys.stdin, help = 'Path to network to be visualized.')
parser.add_argument('--network2', nargs=1, type= str, default=sys.stdin, help = 'Path to network to be visualized.')
parser.add_argument('--network3', nargs=1, type= str, default=sys.stdin, help = 'Path to network to be visualized.')
parser.add_argument('--network4', nargs=1, type= str, default=sys.stdin, help = 'Path to network to be visualized.')
parser.add_argument('--network5', nargs=1, type= str, default=sys.stdin, help = 'Path to network to be visualized.')

parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to outdir.')

###FUNCTIONS###
def visualize(network, outname):
    '''Vizualize a network
    '''
    #Create a graph
    gr = nx.Graph()
    gr.add_edges_from(network)

    fig, ax = plt.subplots(figsize=(4.5/2.54, 4/2.54))
    nx.draw(gr, node_size=5)

    plt.axis('off')
    fig.tight_layout()
    fig.savefig(outname, format='png', dpi=300)
    plt.close()

    return None
#####MAIN#####
#Set font size
matplotlib.rcParams.update({'font.size': 5.5})
args = parser.parse_args()

network1 = np.load(args.network1[0], allow_pickle=True)
network2 = np.load(args.network1[0], allow_pickle=True)
network3 = np.load(args.network1[0], allow_pickle=True)
network4 = np.load(args.network1[0], allow_pickle=True)
network5 = np.load(args.network1[0], allow_pickle=True)

outdir = args.outdir[0]

pdb.set_trace()
